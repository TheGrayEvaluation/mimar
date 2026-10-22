//! Minimal JSON (oku + yaz), saf std. Beyin dosyasının gövdesi JSON'dur; motor
//! hem okur hem YAZAR (yeni beyin oluşturma + öğrenileni kaydetme).

#[derive(Debug, Clone)]
pub enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    pub fn get(&self, k: &str) -> Option<&J> {
        if let J::Obj(m) = self {
            m.iter().find(|(kk, _)| kk == k).map(|(_, v)| v)
        } else {
            None
        }
    }
    pub fn get_mut(&mut self, k: &str) -> Option<&mut J> {
        if let J::Obj(m) = self {
            m.iter_mut().find(|(kk, _)| kk == k).map(|(_, v)| v)
        } else {
            None
        }
    }
    pub fn set(&mut self, k: &str, v: J) {
        if let J::Obj(m) = self {
            if let Some(slot) = m.iter_mut().find(|(kk, _)| kk == k) {
                slot.1 = v;
            } else {
                m.push((k.to_string(), v));
            }
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let J::Str(s) = self { Some(s) } else { None }
    }
    pub fn as_f64(&self) -> Option<f64> {
        if let J::Num(n) = self { Some(*n) } else { None }
    }
    pub fn as_bool(&self) -> bool {
        matches!(self, J::Bool(true))
    }
    pub fn obj_entries_mut(&mut self) -> Option<&mut Vec<(String, J)>> {
        if let J::Obj(m) = self { Some(m) } else { None }
    }
    pub fn s(v: &str) -> J { J::Str(v.to_string()) }
    pub fn n(v: f64) -> J { J::Num(v) }
}

// ----------------------------- çözücü -------------------------------------
pub fn parse(bytes: &[u8]) -> Result<J, String> {
    let mut p = P { b: bytes, i: 0 };
    p.ws();
    p.value()
}

struct P<'a> { b: &'a [u8], i: usize }

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() {
            match self.b[self.i] {
                b' ' | b'\t' | b'\n' | b'\r' => self.i += 1,
                _ => break,
            }
        }
    }
    fn value(&mut self) -> Result<J, String> {
        self.ws();
        if self.i >= self.b.len() { return Err("beklenmedik son".into()); }
        match self.b[self.i] {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(J::Str(self.string()?)),
            b't' => { self.lit("true")?; Ok(J::Bool(true)) }
            b'f' => { self.lit("false")?; Ok(J::Bool(false)) }
            b'n' => { self.lit("null")?; Ok(J::Null) }
            _ => self.number(),
        }
    }
    fn lit(&mut self, s: &str) -> Result<(), String> {
        if self.b[self.i..].starts_with(s.as_bytes()) { self.i += s.len(); Ok(()) }
        else { Err(format!("sabit beklendi: {}", s)) }
    }
    fn object(&mut self) -> Result<J, String> {
        self.i += 1;
        let mut m = Vec::new();
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == b'}' { self.i += 1; return Ok(J::Obj(m)); }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            if self.b.get(self.i) != Some(&b':') { return Err("':' beklendi".into()); }
            self.i += 1;
            let v = self.value()?;
            m.push((k, v));
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => { self.i += 1; break; }
                _ => return Err("',' veya '}' beklendi".into()),
            }
        }
        Ok(J::Obj(m))
    }
    fn array(&mut self) -> Result<J, String> {
        self.i += 1;
        let mut a = Vec::new();
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == b']' { self.i += 1; return Ok(J::Arr(a)); }
        loop {
            let v = self.value()?;
            a.push(v);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => { self.i += 1; break; }
                _ => return Err("',' veya ']' beklendi".into()),
            }
        }
        Ok(J::Arr(a))
    }
    fn string(&mut self) -> Result<String, String> {
        if self.b.get(self.i) != Some(&b'"') { return Err("'\"' beklendi".into()); }
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        while self.i < self.b.len() {
            let c = self.b[self.i];
            match c {
                b'"' => { self.i += 1; return String::from_utf8(out).map_err(|_| "utf8".to_string()); }
                b'\\' => {
                    self.i += 1;
                    if self.i >= self.b.len() { return Err("kaçış sonrası eksik".into()); }
                    let e = self.b[self.i]; self.i += 1;
                    match e {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0c),
                        b'u' => {
                            if self.i + 4 > self.b.len() { return Err("eksik \\u kaçışı".into()); }
                            let hex = std::str::from_utf8(&self.b[self.i..self.i + 4]).unwrap_or("0000");
                            let cp = u32::from_str_radix(hex, 16).unwrap_or(0);
                            self.i += 4;
                            if let Some(ch) = char::from_u32(cp) {
                                let mut buf = [0u8; 4];
                                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                            }
                        }
                        _ => out.push(e),
                    }
                }
                _ => { out.push(c); self.i += 1; }
            }
        }
        Err("kapanmayan string".into())
    }
    fn number(&mut self) -> Result<J, String> {
        let start = self.i;
        while self.i < self.b.len() {
            match self.b[self.i] {
                b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E' => self.i += 1,
                _ => break,
            }
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "sayı utf8")?;
        s.parse::<f64>().map(J::Num).map_err(|_| format!("geçersiz sayı: {}", s))
    }
}

// ----------------------------- yazıcı -------------------------------------
pub fn write(j: &J) -> String {
    let mut s = String::new();
    put(j, &mut s);
    s
}

fn put(j: &J, out: &mut String) {
    match j {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Num(n) => {
            if n.fract() == 0.0 && n.abs() < 1e15 {
                out.push_str(&format!("{}", *n as i64));
            } else {
                // Beyin boyutu: 6 ondalığa yuvarla + sondaki sıfırları at.
                // vec/ml float'larını ~2x küçültür; cosine/ML'e etkisi ihmal edilebilir,
                // çok küçük ağırlıklar 0'a iner (daha seyrek). (Tam-repr parite feda edilir.)
                let s = format!("{:.6}", n);
                let t = s.trim_end_matches('0').trim_end_matches('.');
                out.push_str(if t.is_empty() || t == "-" { "0" } else { t });
            }
        }
        J::Str(s) => put_str(s, out),
        J::Arr(a) => {
            out.push('[');
            for (i, v) in a.iter().enumerate() {
                if i > 0 { out.push(','); }
                put(v, out);
            }
            out.push(']');
        }
        J::Obj(m) => {
            out.push('{');
            for (i, (k, v)) in m.iter().enumerate() {
                if i > 0 { out.push(','); }
                put_str(k, out);
                out.push(':');
                put(v, out);
            }
            out.push('}');
        }
    }
}

fn put_str(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}
