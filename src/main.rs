#[macro_use]
extern crate glium;
extern crate rusttype;
extern crate rand;
extern crate time;

#[macro_use]
extern crate more_asserts;

#[macro_use]
mod util;

mod math;
mod font;
mod manifold;
mod bsp;

use bsp::{BspNode, Face};
use manifold::Manifold;
use math::*;
use math::pose::Pose;
use glium::{DisplayBuild, Surface};
use std::collections::{HashSet};
use std::cell::{RefCell};
use std::rc::Rc;
use std::f32;
use glium::backend::glutin_backend::GlutinFacade;

// flat phong
static FLAT_VS: &'static str = r#"
#version 140
in vec3 position;
out vec3 v_position, v_viewpos;
uniform mat4 perspective, view, model;
void main() {
    mat4 modelview = view * model;
    vec4 mpos = modelview * vec4(position, 1.0);
    gl_Position = perspective * mpos;
    v_position = gl_Position.xyz / gl_Position.w;
    v_viewpos = -mpos.xyz;
}"#;

static FLAT_FS: &'static str = r#"
#version 140
in vec3 v_position, v_viewpos;
out vec4 color;
uniform vec3 u_light;
uniform vec4 u_color;
const vec3 ambient_color = vec3(0.1, 0.1, 0.1);
const vec3 specular_color = vec3(1.0, 1.0, 1.0);
void main() {
    vec3 normal = normalize(cross(dFdx(v_viewpos), dFdy(v_viewpos)));
    float diffuse = abs(dot(normal, normalize(u_light))); // ...
    vec3 camera_dir = normalize(-v_position);
    vec3 half_direction = normalize(normalize(u_light) + camera_dir);
    float specular = pow(max(dot(half_direction, normal), 0.0), 16.0);
    color = vec4(ambient_color + diffuse*u_color.rgb + specular * specular_color, u_color.a);
}"#;

// solid color
static SOLID_VS: &'static str = r#"
#version 140
in vec3 position;
uniform mat4 perspective, view, model;
void main() {
    mat4 modelview = view * model;
    vec4 mpos = modelview * vec4(position, 1.0);
    gl_Position = perspective * mpos;
}"#;

static SOLID_FS: &'static str = r#"
#version 140
out vec4 color;
uniform vec4 u_color;
void main() {
    color = u_color;
}"#;


fn unpack_arrays<'a>(arrays: &'a [[u16; 3]]) -> &'a [u16] {
    unsafe { std::slice::from_raw_parts(arrays.as_ptr() as *const u16, arrays.len() * 3) }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3]
}

implement_vertex!(Vertex, position);


fn vertex_slice<'a>(v3s: &'a [V3]) -> &'a [Vertex] {
    unsafe { std::slice::from_raw_parts(v3s.as_ptr() as *const Vertex, v3s.len()) }
}

struct InputState {
    pub mouse_pos: (i32, i32),
    pub mouse_pos_prev: (i32, i32),
    pub mouse_vec: V3,
    pub mouse_vec_prev: V3,
    pub view_angle: f32,
    pub size: (u32, u32),
    pub mouse_down: bool,
    pub keys_down: HashSet<glium::glutin::VirtualKeyCode>,
    pub key_changes: Vec<(glium::glutin::VirtualKeyCode, bool)>
}

impl InputState {
    fn new(w: u32, h: u32, view_angle: f32) -> InputState {
        InputState {
            mouse_pos: (0, 0),
            mouse_pos_prev: (0, 0),
            mouse_vec: V3::zero(),
            mouse_vec_prev: V3::zero(),
            view_angle: view_angle,
            size: (w, h),
            mouse_down: false,
            keys_down: HashSet::new(),
            key_changes: Vec::new(),
        }
    }

    fn get_projection_matrix(&self, near: f32, far: f32) -> M4x4 {
        M4x4::perspective(self.view_angle.to_radians(), self.size.0 as f32 / self.size.1 as f32, near, far)
    }

    pub fn mouse_delta(&self) -> V2 {
        let now = vec2(self.mouse_pos.0 as f32, self.mouse_pos.1 as f32);
        let prev = vec2(self.mouse_pos_prev.0 as f32, self.mouse_pos_prev.1 as f32);
        (now - prev)
    }

    #[inline]
    pub fn mouse_pos_v(&self) -> V2 {
        vec2(self.mouse_pos.0 as f32, self.mouse_pos.1 as f32)
    }

    #[inline]
    pub fn mouse_prev_v(&self) -> V2 {
        vec2(self.mouse_pos_prev.0 as f32, self.mouse_pos_prev.1 as f32)
    }

    #[inline]
    pub fn dims(&self) -> V2 {
        vec2(self.size.0 as f32, self.size.1 as f32)
    }

    #[inline]
    pub fn scaled_mouse_delta(&self) -> V2 {
        (self.mouse_pos_v() - self.mouse_prev_v()) / self.dims()*0.5
    }

    #[inline]
    pub fn keys_dir(&self, k1: glium::glutin::VirtualKeyCode, k2: glium::glutin::VirtualKeyCode) -> f32 {
        let v1 = if self.keys_down.contains(&k1) { 1 } else { 0 };
        let v2 = if self.keys_down.contains(&k2) { 1 } else { 0 };
        (v2 - v1) as f32
    }

    fn update(&mut self, display: &glium::backend::glutin_backend::GlutinFacade) -> bool {
        use glium::glutin::{Event, ElementState, MouseButton};
        let mouse_pos = self.mouse_pos;
        let mouse_vec = self.mouse_vec;
        self.key_changes.clear();
        self.mouse_pos_prev = mouse_pos;
        self.mouse_vec_prev = mouse_vec;
        for ev in display.poll_events() {
            match ev {
                Event::Closed => return false,
                Event::Resized(w, h) => {
                    self.size = (w, h);
                },
                Event::Focused(true) => {
                    self.keys_down.clear()
                },
                Event::KeyboardInput(pressed, _, Some(vk)) => {
                    let was_pressed = match pressed {
                        ElementState::Pressed => {
                            self.keys_down.insert(vk);
                            true
                        },
                        ElementState::Released => {
                            self.keys_down.remove(&vk);
                            false
                        }
                    };
                    self.key_changes.push((vk, was_pressed));
                },
                Event::MouseMoved(x, y) => {
                    self.mouse_pos = (x, y);
                },
                Event::MouseInput(state, MouseButton::Left) => {
                    self.mouse_down = state == ElementState::Pressed;
                }
                _ => (),
            }
        }
        self.mouse_vec = {
            let spread = (self.view_angle.to_radians() * 0.5).tan();
            let (w, h) = (self.size.0 as f32, self.size.1 as f32);
            let (mx, my) = (self.mouse_pos.0 as f32, self.mouse_pos.1 as f32);
            let hh = h * 0.5;
            let y = spread * (h - my - hh) / hh;
            let x = spread * (mx - w * 0.5) / hh;
            vec3(x, y, -1.0).normalize().unwrap()
        };
        true
    }
}

pub fn glmat(inp: M4x4) -> [[f32; 4]; 4] { inp.into() }

struct DemoWindow {
    pub display: GlutinFacade,
    pub input: InputState,
    pub view: M4x4,
    pub lit_shader: glium::Program,
    pub solid_shader: glium::Program,
    pub clear_color: V4,
    pub light_pos: [f32; 3],
    pub targ: Option<glium::Frame>,
    pub near_far: (f32, f32),
    pub dt: f64,
    pub mspf: f32,
    pub fps: f32,
    pub frame: u64,
    frame_start: u64,
    last_fps_time: u64,
}

impl DemoWindow {
    pub fn new() -> DemoWindow {
        let display = glium::glutin::WindowBuilder::new()
                            .with_depth_buffer(24)
                            .with_vsync()
                            .build_glium()
                            .unwrap();
        let input_state = {
            let (win_w, win_h) = display.get_window().unwrap()
                .get_inner_size_pixels().unwrap();
            InputState::new(win_w, win_h, 75.0)
        };

        let phong_program = glium::Program::from_source(&display, FLAT_VS, FLAT_FS, None).unwrap();
        let solid_program = glium::Program::from_source(&display, SOLID_VS, SOLID_FS, None).unwrap();
        DemoWindow {
            display: display,
            input: input_state,
            view: M4x4::identity(),
            lit_shader: phong_program,
            solid_shader: solid_program,
            clear_color: vec4(0.5, 0.6, 1.0, 1.0),
            light_pos: [1.4, 0.4, 0.7f32],
            near_far: (0.01, 500.0),
            targ: None,
            frame: 0,
            fps: 60.0,
            dt: 1.0 / 60.0,
            mspf: 0.0,
            frame_start: time::precise_time_ns(),
            last_fps_time: time::precise_time_ns(),
        }
    }

    pub fn up(&mut self) -> bool {
        assert!(self.targ.is_none());
        let last_frame_start = self.frame_start;
        self.frame_start = time::precise_time_ns();
        let delta_ns = (self.frame_start - last_frame_start) as f64;
        self.dt = delta_ns * 1.0e-9;
        self.fps = (1.0 / self.dt) as f32;

        if !self.input.update(&self.display) {
            false
        } else {
            self.targ = Some(self.display.draw());
            self.targ.as_mut().unwrap().clear_color_and_depth(self.clear_color.into(), 1.0);
            true
        }
    }


    pub fn draw_lit_tris(&mut self, mat: M4x4, color: V4, verts: &[V3], maybe_tris: Option<&[[u16; 3]]>) {
        let vbo = glium::VertexBuffer::new(&self.display, vertex_slice(verts)).unwrap();
        let params = glium::DrawParameters {
            blend: glium::Blend::alpha_blending(),
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            .. Default::default()
        };
        let uniforms = uniform! {
            model: glmat(mat),
            u_color: <[f32; 4]>::from(color),
            view: glmat(self.view),
            perspective: glmat(self.input.get_projection_matrix(self.near_far.0, self.near_far.1)),
            u_light: self.light_pos,
        };
        if let Some(tris) = maybe_tris {
            let ibo = glium::IndexBuffer::new(&self.display,
                glium::index::PrimitiveType::TrianglesList, unpack_arrays(tris)).unwrap();
            self.targ.as_mut().unwrap().draw((&vbo,), &ibo, &self.lit_shader, &uniforms, &params).unwrap();
        } else {
            let ibo = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);
            self.targ.as_mut().unwrap().draw((&vbo,), &ibo, &self.lit_shader, &uniforms, &params).unwrap();
        }
    }

    pub fn draw_solid(&mut self, mat: M4x4, color: V4, verts: &[V3], prim_type: glium::index::PrimitiveType) {
        let vbo = glium::VertexBuffer::new(&self.display, vertex_slice(verts)).unwrap();
        let ibo = glium::index::NoIndices(prim_type);

        self.targ.as_mut().unwrap().draw((&vbo,), &ibo, &self.solid_shader,
                &uniform! {
                    model: glmat(mat),
                    u_color: <[f32; 4]>::from(color),
                    view: glmat(self.view),
                    perspective: glmat(self.input.get_projection_matrix(
                        self.near_far.0, self.near_far.1)),
                },
                &glium::DrawParameters {
                    point_size: Some(5.0),
                    blend: glium::Blend::alpha_blending(),
                    .. Default::default()
                }).unwrap();
    }

    pub fn wf_manifold(&mut self, mat: M4x4, color: V4, mf: &Manifold) {
        let mut verts = Vec::with_capacity(mf.edges.len()*2);
        for e in &mf.edges {
            verts.push(mf.verts[e.vert_idx()]);
            verts.push(mf.verts[mf.edges[e.next_idx()].vert_idx()]);
        }
        self.draw_solid(mat, color, &verts[..], glium::index::PrimitiveType::LinesList);
    }

    pub fn draw_face(&mut self, mat: M4x4, color: V4, f: &Face) {
        self.draw_lit_tris(mat, color, &f.vertex, Some(&f.gen_tris()[..]));
    }

    pub fn draw_faces(&mut self, mat: M4x4, faces: &[Face]) {
        for face in faces.iter() {
            self.draw_face(mat, V4::expand(face.plane.normal, 1.0), face);
        }
    }

    pub fn end_frame(&mut self) {
        let now = time::precise_time_ns();
        self.mspf = ((now - self.frame_start) as f32) / 1.0e6;
        self.targ.take().unwrap().finish().unwrap();
    }
}

pub fn run_bsp_test() {
    let font_data = include_bytes!("../data/Arial Unicode.ttf");

    let mut win = DemoWindow::new();
    let mut fr = font::FontRenderer::new(&mut win.display, &font_data[..]);

    win.input.view_angle = 45.0;
    let mut draw_mode = 0;
    let mut drag_mode = 1;
    let mut cam = Pose::from_rotation(Quat::from_axis_angle(vec3(1.0, 0.0, 0.0), 60f32.to_radians()));
    let mut cam_dist = 5_f32;
    let mut hit_dist = 0_f32;
    win.light_pos = [-1.0, 0.5, 0.5];

    let mut bpos = vec3(0.0, 0.0, 0.5);
    let mut cpos = vec3(0.8, 0.0, 0.45);

    let ac = Manifold::new_cube(1.0);
    let bc = Manifold::new_box(vec3(-0.5, -0.5, -1.2), vec3(0.5, 0.5, 1.2));
    let co = Manifold::new_cube(1.0).dual_r(0.85);

    let af = ac.faces();
    let bf = bc.faces();
    let cf = co.faces();

    let mut bsp = None;

    let mut faces: Vec<Face> = Vec::new();

    win.near_far = (0.01, 10.0);


    while win.up() {
        {
            let (screen_width, screen_height) = {
                let (w, h) = win.display.get_framebuffer_dimensions();
                (w as f32, h as f32)
            };
            fr.begin_frame(screen_width, screen_height);
        }
        if win.input.key_changes.iter().any(|&(a, b)| b && a == glium::glutin::VirtualKeyCode::D) {
            draw_mode = (draw_mode+1)%2;
        }

        if win.input.mouse_down {
            match drag_mode {
                1 => {
                    cam.orientation *= Quat::virtual_track_ball(vec3(0.0, 0.0, 2.0), V3::zero(), win.input.mouse_vec_prev, win.input.mouse_vec).conj();
                },
                0 => {
                    drag_mode = 1;
                    let v0 = cam.position;
                    let v1 = cam.position + cam.orientation * (win.input.mouse_vec*100.0);
                    let bhit = geom::convex_hit_check_posed(
                        &bc.faces[..], Pose::from_translation(bpos), v0, v1);
                    let v1 = bhit.impact;
                    let chit = geom::convex_hit_check_posed(
                        &co.faces[..], Pose::from_translation(cpos), v0, v1);
                    hit_dist = v0.dist(chit.impact);
                    if bhit.did_hit {
                        drag_mode = 2
                    }
                    if chit.did_hit {
                        drag_mode = 3;
                    }
                    if draw_mode == 2 {
                        drag_mode = 1;
                    }
                    println!("DRAG MODE => {}", drag_mode);
                },
                n => {
                    let pos = if n == 2 { &mut bpos } else { &mut cpos };
                    *pos += (cam.orientation * win.input.mouse_vec - cam.orientation * win.input.mouse_vec_prev) * hit_dist;
                    bsp = None;
                },
            }
        } else {
            drag_mode = 0;
        }

        cam.position = cam.orientation.z_dir() * cam_dist;
        if bsp.is_none() {
            let mut bsp_a = Box::new(bsp::compile(af.clone(), Manifold::new_cube(2.0)));
            let mut bsp_b = Box::new(bsp::compile(bf.clone(), Manifold::new_cube(2.0)));
            let mut bsp_c = Box::new(bsp::compile(cf.clone(), Manifold::new_cube(2.0)));

            bsp_b.translate(bpos);
            bsp_c.translate(cpos);

            bsp_b.negate();
            bsp_c.negate();

            let mut bspres = bsp::intersect(bsp_c, bsp::intersect(bsp_b, bsp_a));

            let brep = bspres.rip_brep();
            bspres.make_brep(brep, 0);

            faces = bspres.rip_brep();
            bsp = bsp::clean(bspres);
            assert!(bsp.is_some());
        }

        win.view = cam.inverse().to_mat4();

        win.wf_manifold(M4x4::identity(), vec4(0.0, 1.0, 0.5, 1.0), &ac);
        win.wf_manifold(M4x4::from_translation(bpos), vec4(0.0, 0.5, 1.0, 1.0), &bc);
        win.wf_manifold(M4x4::from_translation(cpos), vec4(0.5, 0.0, 1.0, 1.0), &co);


        fr.write((1, 1), format!("FPS: {}", win.fps).as_str());

        match draw_mode {
            0 => {
                // faces (boundary)
                win.draw_faces(M4x4::identity(), &faces[..]);
            },
            1 => {
                // cells
                let mut stack = vec![bsp.as_ref().unwrap().as_ref()];
                while let Some(n) = stack.pop() {
                    if n.leaf_type == bsp::LeafType::Under {
                        let c = n.convex.verts.iter().fold(V3::zero(), |a, &b| a+b) / (n.convex.verts.len() as f32);
                        let mut m = M4x4::from_translation(c);
                        m *= M4x4::from_scale(V3::splat(0.95));
                        m *= M4x4::from_translation(-c);
                        win.draw_lit_tris(m, V4::splat(1.0), &n.convex.verts[..], Some(&n.convex.generate_tris()[..]))
                    }
                    if let Some(ref r) = n.under {
                        stack.push(r.as_ref());
                    }
                    if let Some(ref r) = n.over {
                        stack.push(r.as_ref());
                    }
                }
            },
            _ => {
                unreachable!("bad draw_mode {}", draw_mode);
            }
        }

        fr.draw(&mut win.display, win.targ.as_mut().unwrap());

        win.end_frame();
    }
}

fn main() {
    run_bsp_test();
}
