use math::*;
use glium;
use glium::backend::glutin_backend::GlutinFacade;
use rusttype::{FontCollection, Font, Scale, point, vector, PositionedGlyph};
use rusttype::gpu_cache::{Cache};
use rusttype::Rect;

use std::borrow::Cow;

// mostly the rusttype example code...

#[derive(Clone, Copy, Debug, Default)]
struct FontVertex {
    position: [f32; 2],
    texcoord: [f32; 2],
    color: [f32; 4],
}

implement_vertex!(FontVertex, position, texcoord, color);
static FONT_VS: &'static str = r#"
#version 140
in vec2 position;
in vec2 texcoord;
in vec4 color;

out vec2 v_texcoord;
out vec4 v_color;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
    v_color = color;
}"#;
static FONT_FS: &'static str = r#"
#version 140
uniform sampler2D u_font;
in vec2 v_texcoord;
in vec4 v_color;
out vec4 color;
void main() {
    color = v_color * vec4(1.0, 1.0, 1.0, texture(u_font, v_texcoord).r);
}"#;


pub struct FontRenderer<'a> {
    cache: Cache,
    font: Font<'a>,
    program: glium::Program,
    cache_tex: glium::texture::Texture2d,
    verts: Vec<FontVertex>,
    dpi: f32,
    screen: (f32, f32),
    pub color: V4,
    pub scale: Scale
}

impl<'a> FontRenderer<'a> {
    pub fn new(display: &mut GlutinFacade, font_bytes: &'a [u8]) -> FontRenderer<'a> {
        let font = FontCollection::from_bytes(font_bytes).into_font().unwrap();
        let dpi_factor = display.get_window().unwrap().hidpi_factor();

        let (cache_width, cache_height) = (512 * dpi_factor as u32, 512 * dpi_factor as u32);
        let mut cache = Cache::new(cache_width, cache_height, 0.1, 0.1);

        let program = program!(display, 140 => { vertex: FONT_VS, fragment: FONT_FS }).unwrap();
        let cache_tex = glium::texture::Texture2d::with_format(display,
            glium::texture::RawImage2d {
                data: Cow::Owned(vec![128u8; cache_width as usize * cache_height as usize]),
                width: cache_width,
                height: cache_height,
                format: glium::texture::ClientFormat::U8
            },
            glium::texture::UncompressedFloatFormat::U8,
            glium::texture::MipmapsOption::NoMipmap).unwrap();
        let screen = {
            let (w, h) = display.get_framebuffer_dimensions();
            (w as f32, h as f32)
        };

        FontRenderer {
            font: font,
            cache: cache,
            program: program,
            cache_tex: cache_tex,
            dpi: dpi_factor,
            screen: screen,
            verts: Vec::new(),
            color: V4::splat(1.0),
            scale: Scale::uniform(24.0*dpi_factor)
        }
    }

    pub fn begin_frame(&mut self, screen_w: f32, screen_h: f32) {
        self.screen = (screen_w, screen_h);
        self.verts.clear();
    }

    pub fn write(&mut self, pos: (isize, isize), text: &str) {
        let fpos = ((pos.0 as f32) * self.scale.x, (pos.1 as f32) * self.scale.y*1.2);
        self.write_f(fpos, text)
    }

    pub fn write_f(&mut self, pos: (f32, f32), text: &str) {
        let glyphs = self.font.layout(text, self.scale, point(pos.0, pos.1)).collect::<Vec<_>>();
        for glyph in &glyphs {
            self.cache.queue_glyph(0, glyph.clone());
        }
        let cache_tex = &mut self.cache_tex;
        self.cache.cache_queued(|rect, data| {
            cache_tex.main_level().write(glium::Rect {
                left: rect.min.x,
                bottom: rect.min.y,
                width: rect.width(),
                height: rect.height()
            }, glium::texture::RawImage2d {
                data: Cow::Borrowed(data),
                width: rect.width(),
                height: rect.height(),
                format: glium::texture::ClientFormat::U8
            });
        }).unwrap();
        let origin = point(0.0, 0.0);
        for glyph in &glyphs {
            if let Ok(Some((uv_rect, screen_rect))) = self.cache.rect_for(0, glyph) {
                let gl_rect = Rect {
                    min: origin
                        + (vector(screen_rect.min.x as f32 / self.screen.0 - 0.5,
                                  1.0 - screen_rect.min.y as f32 / self.screen.1 - 0.5)) * 2.0,
                    max: origin
                        + (vector(screen_rect.max.x as f32 / self.screen.0 - 0.5,
                                  1.0 - screen_rect.max.y as f32 / self.screen.1 - 0.5)) * 2.0
                };
                self.verts.reserve(6);
                let color = self.color.into();
                self.verts.push(FontVertex {
                    position: [gl_rect.min.x, gl_rect.max.y],
                    texcoord: [uv_rect.min.x, uv_rect.max.y],
                    color: color
                });
                self.verts.push(FontVertex {
                    position: [gl_rect.min.x, gl_rect.min.y],
                    texcoord: [uv_rect.min.x, uv_rect.min.y],
                    color: color
                });
                self.verts.push(FontVertex {
                    position: [gl_rect.max.x, gl_rect.min.y],
                    texcoord: [uv_rect.max.x, uv_rect.min.y],
                    color: color
                });
                self.verts.push(FontVertex {
                    position: [gl_rect.max.x, gl_rect.min.y],
                    texcoord: [uv_rect.max.x, uv_rect.min.y],
                    color: color
                });
                self.verts.push(FontVertex {
                    position: [gl_rect.max.x, gl_rect.max.y],
                    texcoord: [uv_rect.max.x, uv_rect.max.y],
                    color: color
                });
                self.verts.push(FontVertex {
                    position: [gl_rect.min.x, gl_rect.max.y],
                    texcoord: [uv_rect.min.x, uv_rect.max.y],
                    color: color
                });
            }
        }
    }

    pub fn draw<S>(&mut self, display: &mut GlutinFacade, target: &mut S)
            where S: glium::Surface {
        let vbo = glium::VertexBuffer::new(display, &self.verts[..]).unwrap();
        target.draw(&vbo,
            glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
            &self.program,
            &uniform!{
                u_font: self.cache_tex.sampled()
                    .magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest)
            },
            &glium::DrawParameters {
                blend: glium::Blend::alpha_blending(),
                ..Default::default()
            }
        ).unwrap()
    }

}
