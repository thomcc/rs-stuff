use math::*;
use math::geom::*;
use manifold::Manifold;
use util::OrdFloat;
use std::{f32, mem};

const Q_SNAP: f32 = 0.05;
const QUANTIZE_CHECK: f32 = Q_SNAP * (1.0 / 256.0 * 0.5);
const FUZZY_WIDTH: f32 = 100.0*DEFAULT_PLANE_WIDTH;

const ALLOW_AXIAL: u8 = 0b001;
const FACE_TEST_LIMIT: usize = 50;

const OVER: usize = PlaneTestResult::Over as usize;
const UNDER: usize = PlaneTestResult::Under as usize;
const SPLIT: usize = PlaneTestResult::Split as usize;
const COPLANAR: usize = PlaneTestResult::Coplanar as usize;


#[derive(Clone, Default)]
pub struct Face {
    pub plane: Plane,
    pub mat_id: usize,
    pub vertex: Vec<V3>,
    pub gu: V3,
    pub gv: V3,
    pub ot: V3,
}

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum LeafType {
    NotLeaf = 0b00,
    Under = (PlaneTestResult::Under as u8), // 0b01
    Over = (PlaneTestResult::Over as u8), // 0b10
}

default_for_enum!(LeafType::NotLeaf);

#[derive(Clone, Default)]
pub struct BspNode {
    pub plane: Plane,
    pub under: Option<Box<BspNode>>,
    pub over: Option<Box<BspNode>>,
    pub leaf_type: LeafType,
    pub convex: Manifold,
    pub boundary: Vec<Face>,
}

#[derive(Clone)]
pub struct BspPreorder<'a> {
    stack: Vec<&'a BspNode>
}

impl<'a> Iterator for BspPreorder<'a> {
    type Item = &'a BspNode;
    fn next(&mut self) -> Option<&'a BspNode> {
        let node = try_opt!(self.stack.pop());
        if let Some(ref b) = node.under {
            self.stack.push(b.as_ref()); // is &*b the same?
        }
        if let Some(ref b) = node.over {
            self.stack.push(b.as_ref());
        }
        Some(node)
    }
}

#[derive(Clone)]
pub struct BspBackToFront<'a> {
    pub p: V3,
    stack: Vec<&'a BspNode>,
}

impl<'a> Iterator for BspBackToFront<'a> {
    type Item = &'a BspNode;
    fn next(&mut self) -> Option<&'a BspNode> {
        let node = try_opt!(self.stack.pop());
        let plane = Plane::new(self.p, 1.0);
        let mut np = if plane.dot(node.plane) > 0.0 {
            &node.over
        } else {
            &node.under
        };
        while let &Some(ref n) = np {
            self.stack.push(n.as_ref());
            if plane.dot(n.plane) > 0.0 {
                np = &n.over;
            } else {
                np = &n.under;
            }
        }
        Some(node)
    }
}

fn plane_cost_c(input: &[Face], split: Plane, space: &Manifold) -> (f32, [f32; 4]) {
    let mut counts = [0.0f32, 0.0, 0.0, 0.0];
    for face in input.iter() {
        counts[face.split_test_val(split, FUZZY_WIDTH)] += 1.0;
    }
    if space.verts.is_empty() {
        return (((counts[OVER]-counts[UNDER]).abs() + counts[SPLIT] - counts[COPLANAR]), counts)
    }

    let vol_total = space.volume();
    let space_under = space.cropped(split);
    let space_over = space.cropped(-split);

    let vol_over = space_over.volume();
    let vol_under = space_under.volume();

    assert_ge!(vol_over / vol_total, -0.01);
    assert_ge!(vol_under / vol_total, -0.01);

    (vol_over * (counts[OVER] + 1.5*counts[SPLIT]).powf(0.9) +
     vol_under * (counts[UNDER] + 1.5*counts[SPLIT]).powf(0.9),
     counts)
}

fn plane_cost(input: &[Face], split: Plane, space: &Manifold) -> f32 {
    plane_cost_c(input, split, space).0
}

pub fn compile(faces: Vec<Face>, space: Manifold) -> BspNode {
    compile_lt(faces, space, LeafType::NotLeaf)
}

pub fn compile_lt(mut faces: Vec<Face>, space: Manifold, side: LeafType) -> BspNode {
    if faces.is_empty() {
        return BspNode {
            convex: space,
            leaf_type: side,
            .. BspNode::new(plane(0.0, 0.0, 0.0, 0.0))
        };
    }

    faces.sort_by_key(|a| OrdFloat(a.area()));

    let mut min_val = f32::MAX;

    let mut split = Plane::new(V3::zero(), 0.0);

    for (i, face) in faces.iter().enumerate() {
        if i > FACE_TEST_LIMIT { break; }
        let val = plane_cost(&faces[..], face.plane, &space);
        if val < min_val {
            min_val = val;
            split = faces[i].plane;
        }
    }

    assert_ne!(split.normal, V3::zero());

    if ALLOW_AXIAL != 0 && faces.len() > 8 {
        for face in faces.iter() {
            for v in face.vertex.iter() {
                for c in 0..3 {
                    let mask = 1u8 << c;
                    if (ALLOW_AXIAL & mask) != 0 {
                        let mut n = V3::zero(); n[c] = 1.0;
                        let (val, count) = plane_cost_c(&faces[..], Plane::new(n, -v[c]), &space);
                        if val < min_val && (count[OVER] * count[UNDER] > 0.0 || count[SPLIT] > 0.0) {
                            min_val = val;
                            split = Plane::new(n, -v[c]);
                        }
                    }
                }
            }
        }
    }

    let mut node = BspNode::new(split);
    let over_space = space.cropped(-split);
    let under_space = space.cropped(split);

    node.convex = space;

    let (under, over, _) = divide_polys(split, faces);

    for face in over.iter() {
        for v in face.vertex.iter() {
            debug_assert_ge!(dot(node.plane.normal, *v) + node.plane.offset, -FUZZY_WIDTH);
        }
    }
    for face in under.iter() {
        for v in face.vertex.iter() {
            debug_assert_le!(dot(node.plane.normal, *v) + node.plane.offset, FUZZY_WIDTH);
        }
    }

    node.over = Some(Box::new(compile_lt(over, over_space, LeafType::Over)));
    node.under = Some(Box::new(compile_lt(under, under_space, LeafType::Under)));
    node
}

fn gen_faces_rev(mani: &Manifold, mat: usize) -> Vec<Face> {
    let mut r = Vec::with_capacity(mani.faces.len());
    for (i, &mf) in mani.faces.iter().enumerate() {
        let mut f: Face = Default::default();
        f.plane = -mf;
        f.mat_id = mat;
        let e0 = mani.fback[i] as usize;
        let mut e = e0;
        loop {
            f.vertex.push(mani.verts[mani.edges[e].vert_idx()]);
            e = mani.edges[e].prev_idx();
            if e == e0 {
                break;
            }
        }
        f.assign_tex();
        r.push(f);
    }
    r
}

fn gen_faces_mani(mani: &Manifold, mat: usize) -> Vec<Face> {
    let mut r = Vec::with_capacity(mani.faces.len());
    for (i, &mf) in mani.faces.iter().enumerate() {
        let mut f: Face = Default::default();
        f.plane = mf;
        f.mat_id = mat;
        let e0 = mani.fback[i] as usize;
        let mut e = e0;
        loop {
            f.vertex.push(mani.verts[mani.edges[e].vert_idx()]);
            e = mani.edges[e].next_idx();
            if e == e0 {
                break;
            }
        }
        f.assign_tex();
        r.push(f);
    }
    r
}


pub fn divide_polys(split: Plane, input: Vec<Face>) -> (Vec<Face>, Vec<Face>, Vec<Face>) {
    let mut under = Vec::new();
    let mut over = Vec::new();
    let mut coplanar = Vec::new();

    for face in input.into_iter().rev() {
        match face.split_test(split, FUZZY_WIDTH) {
            PlaneTestResult::Coplanar => { coplanar.push(face); },
            PlaneTestResult::Over => { over.push(face); },
            PlaneTestResult::Under => { under.push(face); },
            PlaneTestResult::Split => {
                over.push(face.clip(-split));
                under.push(face.clip(split));
            },
        }
    }
    (under, over, coplanar)
}

impl BspNode {
    pub fn new(plane: Plane) -> BspNode {
        BspNode {
            leaf_type: LeafType::NotLeaf,
            plane: plane,
            .. Default::default()
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.leaf_type != LeafType::NotLeaf
    }

    pub fn only_planes(&self) -> BspNode {
        BspNode {
            leaf_type: self.leaf_type,
            plane: self.plane,
            under: self.under.as_ref().map(|n| Box::new(n.only_planes())),
            over: self.over.as_ref().map(|n| Box::new(n.only_planes())),
            .. Default::default()
        }
    }

    pub fn count(&self) -> usize {
        1 + self.over.as_ref().map_or(0, |node| node.count())
          + self.under.as_ref().map_or(0, |node| node.count())
    }

    pub fn assign_tex(&mut self, mat_id: usize) -> &mut BspNode {
        for face in self.boundary.iter_mut() {
            face.mat_id = mat_id;
            face.assign_tex();
        }
        if let Some(ref mut u) = self.under {
            u.assign_tex(mat_id);
        }
        if let Some(ref mut o) = self.over {
            o.assign_tex(mat_id);
        }
        self
    }

    pub fn derive_convex(&mut self, m: Manifold) {
        if !m.edges.is_empty() && !m.verts.is_empty() {
            assert!(!m.verts.is_empty());
            assert!(!m.edges.is_empty());
            assert!(!m.faces.is_empty());
        }
        self.convex = m;
        if self.is_leaf() {
            return;
        }
        let mut cu = Manifold::new();
        let mut co = Manifold::new();
        if !self.convex.verts.is_empty() {
            match self.convex.split_test(self.plane) {
                PlaneTestResult::Split => {
                    cu = self.convex.cropped( self.plane);
                    co = self.convex.cropped(-self.plane);
                },
                PlaneTestResult::Over => { co = self.convex.clone(); },
                PlaneTestResult::Under => { cu = self.convex.clone(); },
                PlaneTestResult::Coplanar => { unreachable!("has 0 volume somehow?") }
            }
        }
        self.under.as_mut().unwrap().derive_convex(cu);
        self.over.as_mut().unwrap().derive_convex(co);
    }

    pub fn iter(&self) -> BspPreorder {
        BspPreorder { stack: vec![self] }
    }

    pub fn iter_back_to_front(&self, p: V3) -> BspBackToFront {
        BspBackToFront { stack: vec![self], p: p }
    }

    fn embed_face(&mut self, f: Face) {
        if self.leaf_type == LeafType::Over {
            return;
        }
        if self.leaf_type == LeafType::Under {
            self.boundary.push(f);
            return;
        }
        match f.split_test(self.plane, geom::DEFAULT_PLANE_WIDTH) {
            PlaneTestResult::Under => { self.under.as_mut().unwrap().embed_face(f); },
            PlaneTestResult::Over => { self.over.as_mut().unwrap().embed_face(f); },
            PlaneTestResult::Coplanar => {
                if dot(self.plane.normal, f.plane.normal) > 0.0 {
                    self.under.as_mut().unwrap().embed_face(f);
                } else {
                    self.over.as_mut().unwrap().embed_face(f);
                }
            },
            PlaneTestResult::Split => {
                // TODO slice edge here...
                self.over.as_mut().unwrap().embed_face(f.clip(-self.plane));
                self.under.as_mut().unwrap().embed_face(f.clip_self(self.plane));
            }
        }
    }

    fn cut_faces(&self, faces: &mut Vec<Face>) {
        if self.leaf_type == LeafType::Over { return; }
        if self.leaf_type == LeafType::Under { faces.clear(); return; }

        let mut over = Vec::new();
        let mut under = Vec::new();
        let mut planar = Vec::new();

        while let Some(f) = faces.pop() {
            match f.split_test(self.plane, FUZZY_WIDTH) {
                PlaneTestResult::Coplanar => planar.push(f),
                PlaneTestResult::Under => under.push(f),
                PlaneTestResult::Over => over.push(f),
                PlaneTestResult::Split => {
                    under.push(f.clip(self.plane));
                    under.push(f.clip_self(-self.plane));
                }
            }
        }
        self.under.as_ref().unwrap().cut_faces(&mut under);
        self.over.as_ref().unwrap().cut_faces(&mut over);
        faces.reserve(under.len() + over.len() + planar.len());

        faces.append(&mut under);
        faces.append(&mut over);
        faces.append(&mut planar);
    }

    pub fn translate(&mut self, offset: V3) -> &mut BspNode {
        self.plane = self.plane.translate(offset);
        self.convex.translate(offset);
        for face in self.boundary.iter_mut() { face.translate(offset); }
        if self.under.is_some() { self.under.as_mut().unwrap().translate(offset); }
        if self.over.is_some() { self.over.as_mut().unwrap().translate(offset); }
        self
    }

    pub fn rotate(&mut self, rot: Quat) -> &mut BspNode {
        self.plane = self.plane.rotate(rot);
        self.convex.rotate(rot);
        for face in self.boundary.iter_mut() { face.rotate(rot); }
        if self.under.is_some() { self.under.as_mut().unwrap().rotate(rot); }
        if self.over.is_some() { self.over.as_mut().unwrap().rotate(rot); }
        self
    }

    pub fn scale3(&mut self, s: V3) -> &mut BspNode {
        self.plane = self.plane.scale3(s);
        self.convex.scale3(s);
        for face in self.boundary.iter_mut() { face.scale3(s); }
        if self.under.is_some() { self.under.as_mut().unwrap().scale3(s); }
        if self.over.is_some() { self.over.as_mut().unwrap().scale3(s); }
        self
    }


    fn each_mut<F: FnMut(&mut BspNode)>(&mut self, mut f: F) {
        let mut stack: Vec<&mut BspNode> = vec![self];
        while let Some(n) = stack.pop() {
            f(n);
            if n.under.is_some() { stack.push(&mut *n.under.as_mut().unwrap()); }
            if n.over.is_some() { stack.push(&mut *n.over.as_mut().unwrap()) }
        }
    }

    fn splitify_edges(&mut self) -> usize {
        let mut split_count = 0;
        let root = self.only_planes();
        self.each_mut(|n| {
            for face in n.boundary.iter_mut() {
                for j in (0..face.vertex.len()).rev() { split_count += face.edge_splicer(j, &root); }
            }
        });
        split_count
    }

    fn extract_mat(&mut self, face: &Face) {
        for f in self.boundary.iter_mut() { f.extract_mat(face); }
        if self.is_leaf() { return; }
        let f = face.split_test_val(self.plane, FUZZY_WIDTH);
        if f == COPLANAR {
            if dot(self.plane.normal, face.plane.normal) > 0.0 { self.under.as_mut().unwrap().extract_mat(face); }
            else { self.over.as_mut().unwrap().extract_mat(face); }
        } else {
            if (f & UNDER) != 0 { self.under.as_mut().unwrap().extract_mat(face); }
            if (f & OVER) != 0 { self.over.as_mut().unwrap().extract_mat(face); }
        }
    }

    pub fn get_solids(&self) -> Vec<&Manifold> {
        let mut result = Vec::new();
        for n in self.iter() {
            if n.leaf_type == LeafType::Under {
                result.push(&n.convex);
            }
        }
        result
    }

    pub fn make_brep(&mut self, faces: Vec<Face>, mat_id: usize) {
        self.gen_faces(mat_id);
        self.splitify_edges();
        for face in faces {
            self.extract_mat(&face);
        }
    }

    pub fn rip_brep(&mut self) -> Vec<Face> {
        let mut out = Vec::new();
        self.each_mut(|n| {
            out.reserve(n.boundary.len());
            while let Some(f) = n.boundary.pop() {
                out.push(f);
            }
        });
        out
    }

    pub fn negate_tree_planes(&mut self) {
        self.each_mut(|n| {
            for face in n.boundary.iter_mut() {
                face.negate();
            }
            if n.is_leaf() {
                n.leaf_type = if n.leaf_type == LeafType::Under { LeafType::Over } else { LeafType::Under };
            }
            n.plane = -n.plane;
            mem::swap(&mut n.under, &mut n.over);
        });
    }


    fn gen_faces(&mut self, mat_id: usize) {
        let mut to_embed = Vec::new();
        self.each_mut(|n| {
            if n.leaf_type == LeafType::Over {
                to_embed.append(&mut gen_faces_rev(&n.convex, mat_id));
            }
        });
        // self.boundary.reserve(to_embed.len())
        for face in to_embed {
            self.embed_face(face);
        }
    }

    pub fn negate(&mut self) -> &mut BspNode {
        self.negate_tree_planes();
        for f in self.rip_brep() {
            self.embed_face(f);
        }
        self
    }
    // pub fn hit_check(&self, solid: bool, v0: V3, v1: V3) -> Option<V3>;
    // pub fn hit_check_solid_reenter(&self, v0: V3, v1: V3) -> Option<V3>;

}

fn do_union(ao: Option<Box<BspNode>>, mut b: Box<BspNode>) -> Box<BspNode> {
    if ao.is_none() || b.leaf_type == LeafType::Under || ao.as_ref().unwrap().leaf_type == LeafType::Over {
        if ao.is_some() && b.leaf_type == LeafType::Under {
            ao.as_ref().unwrap().cut_faces(&mut b.boundary);
        }
        return b;
    }
    let a = ao.unwrap();

    if a.leaf_type == LeafType::Under || b.leaf_type == LeafType::Over {
        return a;
    }

    assert!(!a.is_leaf());
    assert!(!b.is_leaf());
    let (a_under, a_over) = partition(a, b.plane);
    assert!(a_under.is_some() || a_over.is_some());
    b.under = Some(do_union(a_under, b.under.take().unwrap()));
    b.over = Some(do_union(a_over, b.over.take().unwrap()));
    b
}

pub fn union(a: Box<BspNode>, b: Box<BspNode>) -> Box<BspNode> {
    do_union(Some(a), b)
}

fn do_intersect(ao: Option<Box<BspNode>>, mut b: Box<BspNode>) -> Box<BspNode> {
    if ao.is_none() || b.leaf_type == LeafType::Over || ao.as_ref().unwrap().leaf_type == LeafType::Under {
        if ao.is_some() && ao.as_ref().unwrap().leaf_type == LeafType::Under {
            let mut a = ao.unwrap();
            while let Some(f) = a.boundary.pop() {
                b.embed_face(f);
            }
        }
        return b;
    }
    let mut a = ao.unwrap();
    if b.leaf_type == LeafType::Under || a.leaf_type == LeafType::Over {
        if b.leaf_type == LeafType::Under {
            while let Some(f) = b.boundary.pop() {
                a.embed_face(f);
            }
        }
        return a;
    }

    let (a_under, a_over) = partition(a, b.plane);

    let mut nbu = do_intersect(a_under, b.under.take().unwrap());
    let mut nbo = do_intersect(a_over, b.over.take().unwrap());

    if nbo.is_leaf() && nbo.leaf_type == nbu.leaf_type {
        b.boundary.reserve(nbo.boundary.len() + nbu.boundary.len());
        while let Some(f) = nbo.boundary.pop() {
            b.boundary.push(f);
        }
        while let Some(f) = nbu.boundary.pop() {
            b.boundary.push(f);
        }
        b.leaf_type = nbo.leaf_type;
        b.under = None;
        b.over = None;
    } else {
        b.under = Some(nbu);
        b.over = Some(nbo);
    }
    b
}

pub fn intersect(a: Box<BspNode>, b: Box<BspNode>) -> Box<BspNode> {
    do_intersect(Some(a), b)
}

pub fn clean(mut n: Box<BspNode>) -> Option<Box<BspNode>> {
    if n.convex.verts.len() == 0 {
        return None;
    }

    if n.is_leaf() {
        n.plane = Plane::zero();
        assert!(n.over.is_none());
        assert!(n.under.is_none());
        return Some(n);
    }

    n.over = n.over.take().and_then(clean);
    n.under = n.under.take().and_then(clean);
    match (n.over.is_some(), n.under.is_some()) {
        (false, false) => None,
        (true, false) => n.over,
        (false, true) => n.under,
        (true, true) => {
            let ltu = n.under.as_ref().unwrap().leaf_type;
            let lto = n.over.as_ref().unwrap().leaf_type;
            if lto != LeafType::NotLeaf && ltu == lto {
                n.leaf_type = lto;
                n.plane = Plane::zero();
                n.over = None;
                n.under = None;
            }
            assert!(!n.convex.verts.is_empty());
            Some(n)
        }
    }
}

pub fn partition(mut n: Box<BspNode>, p: Plane) -> (Option<Box<BspNode>>, Option<Box<BspNode>>) {
    match n.convex.split_test(p) {
        PlaneTestResult::Under => {
            return (Some(n), None);
        },
        PlaneTestResult::Over => {
            return (None, Some(n));
        },
        PlaneTestResult::Split => {},
        PlaneTestResult::Coplanar => {
            unreachable!("convex shape is flat?");
        }
    }

    let mut under = Box::new(BspNode::new(n.plane));
    let mut over = Box::new(BspNode::new(n.plane));
    under.leaf_type = n.leaf_type;
    over.leaf_type = n.leaf_type;

    under.convex = n.convex.cropped(p);
    over.convex = n.convex.cropped(-p);
    let (mut under, mut over) =
        if n.leaf_type == LeafType::Under {
            let mut fake = BspNode { under: Some(under), over: Some(over), .. BspNode::new(p) };
            let mut e = Vec::new();
            e.append(&mut n.boundary);
            for face in e.into_iter().rev() {
                fake.embed_face(face)
            }
            assert!(fake.under.is_some());
            assert!(fake.over.is_some());
            (fake.under.take().unwrap(), fake.over.take().unwrap())
        } else {
            (under, over)
        };

    if n.under.is_some() {
        let (uu, ou) = partition(n.under.take().unwrap(), p);
        under.under = uu;
        over.under = ou;
    }

    if n.over.is_some() {
        let (uo, oo) = partition(n.over.take().unwrap(), p);
        under.over = uo;
        over.over = oo;
    }

    if n.is_leaf() {
        assert!(under.is_leaf());
        assert!(over.is_leaf());
        return (Some(under), Some(over));
    }

    assert!(under.over.is_some() || under.under.is_some());
    assert!(over.over.is_some() || over.under.is_some());

    if under.under.is_none() {
        assert!(under.over.is_some());
        under = under.over.take().unwrap();
    }
    else if under.over.is_none() {
        assert!(under.under.is_some());
        under = under.under.take().unwrap();
    }


    if over.under.is_none() {
        assert!(over.over.is_some());
        over = over.over.take().unwrap();
    }
    else if over.over.is_none() {
        assert!(over.under.is_some());
        over = over.under.take().unwrap();
    }
    assert!(under.is_leaf() || (under.under.is_some() && under.over.is_some()));
    assert!(over.is_leaf() || (over.under.is_some() && over.over.is_some()));
    if !under.is_leaf() &&
        (under.over.as_ref().unwrap().is_leaf() &&
         under.over.as_ref().unwrap().leaf_type == under.under.as_ref().unwrap().leaf_type) {
        let mut u = under.under.take().unwrap();
        let mut o = under.over.take().unwrap();
        under.leaf_type = o.leaf_type;
        u.boundary.reverse();
        under.boundary.append(&mut u.boundary);
        o.boundary.reverse();
        under.boundary.append(&mut o.boundary);
    }
    if !over.is_leaf() &&
       over.over.as_ref().unwrap().is_leaf() &&
       over.over.as_ref().unwrap().leaf_type == over.under.as_ref().unwrap().leaf_type {
        let mut u = over.under.take().unwrap();
        let mut o = over.over.take().unwrap();
        over.leaf_type = o.leaf_type;
        u.boundary.reverse();
        over.boundary.append(&mut u.boundary);
        o.boundary.reverse();
        over.boundary.append(&mut o.boundary);
    }
    (Some(under), Some(over))
}


impl Face {
    pub fn new() -> Face { Default::default() }

    pub fn new_quad(v0: V3, v1: V3, v2: V3, v3: V3) -> Face {
        let mut f = Face::new();
        f.vertex = vec![v0, v1, v2, v3];
        let norm = (cross(v1-v0, v2-v1) + cross(v3-v2, v0-v3)).norm_or_unit();
        f.plane = Plane::from_norm_and_point(norm, (v0+v1+v2+v3)*0.25);

        for v in f.vertex.iter() {
            debug_assert_eq!(f.plane.test(*v), PlaneTestResult::Coplanar);
        }
        f.extract_mat_vals(v0, v1, v3, vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0));
        f.gu = f.gu.norm_or_unit();
        f.gv = f.gv.norm_or_v(cross(f.plane.normal, f.gu).norm_or_unit());
        f.ot = V3::zero();
        f
    }

    pub fn new_tri(v0: V3, v1: V3, v2: V3) -> Face {
        let mut f = Face::new();
        f.vertex = vec![v0, v1, v2];
        f.plane = Plane::from_norm_and_point(cross(v1-v0, v2-v1).norm_or_unit(), (v0+v1+v2)*0.25);
        f.gu = (v1-v0).norm_or_unit();
        f.gv = cross(f.plane.normal, f.gu).norm_or_unit();
        f
    }

    pub fn new_tri_tex(v0: V3, v1: V3, v2: V3, t0: V2, t1: V2, t2: V2) -> Face {
        let mut f = Face::new();
        f.vertex = vec![v0, v1, v2];
        f.plane = Plane::from_norm_and_point(cross(v1-v0, v2-v1).norm_or_unit(), (v0+v1+v2)*0.25);
        f.extract_mat_vals(v0, v1, v2, t0, t1, t2);
        f
    }

    pub fn extract_mat_vals(&mut self, v0: V3, v1: V3, v2: V3, t0: V2, t1: V2, t2: V2) {
        self.gu = geom::gradient(v0, v1, v2, t0.x, t1.x, t2.x);
        self.gv = geom::gradient(v0, v1, v2, t0.y, t1.y, t2.y);

        self.ot.x = t0.x - dot(v0, self.gu);
        self.ot.y = t0.y - dot(v0, self.gv);
    }

    pub fn area(&self) -> f32 {
        let mut area = 0.0;
        for i in 2..self.vertex.len() {
            let vb = self.vertex[0];
            let v1 = self.vertex[i-1];
            let v2 = self.vertex[i];
            area += dot(self.plane.normal, cross(v1 - vb, v2 - v1)) * 0.5;
        }
        area
    }

    pub fn center(&self) -> V3 {
        self.vertex.iter().fold(V3::zero(), |a, &b| a + b) * safe_div0(1.0, self.vertex.len() as f32)
    }

    pub fn split_test(&self, plane: Plane, e: f32) -> PlaneTestResult {
        plane.split_test_e(&self.vertex[..], e)
    }

    pub fn split_test_val(&self, plane: Plane, e: f32) -> usize {
        plane.split_test_val_e(&self.vertex[..], e)
    }

    pub fn translate(&mut self, offset: V3) {
        self.plane.translate(offset);
        for v in self.vertex.iter_mut() {
            *v += offset;
        }
        self.ot.x -= dot(offset, self.gu);
        self.ot.y -= dot(offset, self.gv);
    }

    pub fn rotate(&mut self, rot: Quat) {
        self.plane.rotate(rot);
        for v in self.vertex.iter_mut() {
            let r = rot * *v;
            *v = r;
        }
        self.gu = rot * self.gu;
        self.gv = rot * self.gv;
        self.ot = rot * self.ot;
    }

    pub fn scale3(&mut self, s: V3) {
        self.plane.scale3(s);
        for v in self.vertex.iter_mut() {
            let r = s * *v;
            *v = r;
        }
    }

    // point must be interior
    pub fn closest_edge(&self, point: V3) -> usize {
        assert_ge!(self.vertex.len(), 3);
        let mut closest = -1;
        let mut min_d = 0.0;
        for (i, &v0) in self.vertex.iter().enumerate() {
            let i1 = (i+1)% self.vertex.len();
            let v1 = self.vertex[i1];
            let d = line_project(v0, v1, point).dist(point);
            if closest == -1 || d < min_d {
                closest = i as isize;
                min_d = d;
            }
        }
        assert_ge!(closest, 0);
        closest as usize
    }

    pub fn contains_point(&self, s: V3) -> bool {
        for (i, &pp1) in self.vertex.iter().enumerate() {
            let pp2 = self.vertex[(i+1)%self.vertex.len()];
            let side = cross(pp2-pp1, s-pp1);
            if dot(self.plane.normal, side) < 0.0 {
                return false;
            }
        }
        true
    }

    #[inline]
    pub fn vert_uv(&self, i: usize) -> V2 {
        vec2(self.ot.x + dot(self.vertex[i], self.gu),
             self.ot.y + dot(self.vertex[i], self.gv))
    }

    #[inline]
    pub fn uv_at(&self, v: V3) -> V2 {
        vec2(self.ot.x + dot(v, self.gu), self.ot.y + dot(v, self.gv))
    }

    pub fn assign_tex(&mut self) {
        let n = self.plane.normal;
        if n.x.abs() > n.y.abs() && n.x.abs() > n.z.abs() {
            self.gu = vec3(0.0, n.x.signum(), 0.0);
            self.gv = vec3(0.0, 0.0, 1.0);
        } else if n.y.abs() > n.z.abs() {
            self.gu = vec3(-n.y.signum(), 0.0, 0.0);
            self.gv = vec3(0.0, 0.0, 1.0);
        } else {
            self.gu = vec3(1.0, 0.0, 0.0);
            self.gv = vec3(0.0, n.z.signum(), 0.0);
        }
    }

    fn edge_splicer(&mut self, vi0: usize, n: &BspNode) -> usize {
        if n.is_leaf() {
            return 0;
        }
        let mut split_count = 0;
        let vi1 = (vi0 + 1) % self.vertex.len();
        let v0 = self.vertex[vi0];
        let v1 = self.vertex[vi1];
        if v0.dist(v1) <= QUANTIZE_CHECK {
            split_count += 1;
        }
        debug_assert_gt!(v0.dist(v1), QUANTIZE_CHECK);
        let f0 = n.plane.test_e(v0, QUANTIZE_CHECK);
        let f1 = n.plane.test_e(v1, QUANTIZE_CHECK);
        match (f0 as usize)|(f1 as usize) {
            COPLANAR => {
                let count = self.vertex.len();
                split_count += self.edge_splicer(vi0, n.under.as_ref().unwrap());
                let mut k = vi0 + (self.vertex.len() - count);
                while k >= vi0 {
                    split_count += self.edge_splicer(k, n.over.as_ref().unwrap());
                    if k == 0 {
                        break;
                    }
                    k -= 1;
                }
            },
            UNDER => {
                split_count += self.edge_splicer(vi0, n.under.as_ref().unwrap());
            },
            OVER => {
                split_count += self.edge_splicer(vi0, n.over.as_ref().unwrap())
            },
            SPLIT => {
                split_count += 1;
                assert_gt!(v0.dist(v1), QUANTIZE_CHECK);
                let v_mid = n.plane.intersect_with_line(v0, v1);
                assert_gt!(v_mid.dist(v1), QUANTIZE_CHECK);
                assert_gt!(v0.dist(v_mid), QUANTIZE_CHECK);
                assert_eq!(n.plane.test(v_mid), PlaneTestResult::Coplanar);

                self.vertex.insert(vi0 + 1, v_mid);
            },
            _ => {
                unreachable!("Bad plane test result combination? {}", (f0 as usize)|(f1 as usize));
            }
        }
        split_count
    }

    pub fn clip(&self, clip: Plane) -> Face { self.clone().clip_self(clip) }

    pub fn clip_self(mut self, clip: Plane) -> Face {
        debug_assert_eq!(self.split_test(clip, QUANTIZE_CHECK), PlaneTestResult::Split);
        self.slice(clip);
        self.vertex.retain(|&v| clip.test(v) != PlaneTestResult::Over);
        self
    }

    pub fn slice(&mut self, clip: Plane) -> usize {
        let mut c = 0;
        let mut i = 0;
        while i < self.vertex.len() {
            let i2 = (i+1) % self.vertex.len();
            match (clip.test(self.vertex[i]), clip.test(self.vertex[i2])) {
                (PlaneTestResult::Over, PlaneTestResult::Under) |
                (PlaneTestResult::Under, PlaneTestResult::Over) => {
                    let v_mid = clip.intersect_with_line(self.vertex[i], self.vertex[i2]);
                    assert_eq!(clip.test(v_mid), PlaneTestResult::Coplanar);
                    self.vertex.insert(i2, v_mid);
                    i = 0;
                    assert_lt!(c, 2);
                    c += 1;
                },
                _ => {}
            }
            i += 1;
        }
        c
    }

    pub fn negate(&mut self) {
        self.plane = -self.plane;
        self.vertex.reverse();
    }

    fn extract_mat(&mut self, face: &Face) {
        if dot(self.plane.normal, face.plane.normal) < 0.95 {
            return;
        }
        if self.split_test(face.plane, geom::DEFAULT_PLANE_WIDTH) != PlaneTestResult::Coplanar {
            return;
        }
        let interior = face.center();
        if !geom::poly_hit_check(&face.vertex[..], interior+face.plane.normal, interior - face.plane.normal).did_hit {
            return;
        }
        self.mat_id = face.mat_id;
        self.gu = face.gu;
        self.ot = face.ot;
    }

    pub fn gen_tris(&self) -> Vec<[u16; 3]> {
        let mut tris = Vec::with_capacity(self.vertex.len()-2);
        for i in 1..self.vertex.len()-1 {
            tris.push([0, i as u16, (i+1) as u16]);
        }
        tris
    }
}

