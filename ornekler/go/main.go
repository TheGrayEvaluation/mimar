// Mimar — Go örneği (C ABI / cdylib üzerinden cgo ile).
//
// Derleme (C araç zinciri gerekir — Windows'ta mingw/TDM-gcc, Linux'ta gcc):
//
//	cd rust-cekirdek && cargo build --release        # cdylib üretir
//	cd ornekler/go && go run .                        # bu örneği çalıştırır
//
// Felsefe: Mimar YÖNLENDİRİR, host ÇALIŞTIRIR — mimar_resolve tipli bir direktif
// (JSON) döndürür; modeli/aracı çağırmak host'un (bu programın) işidir.
package main

/*
#cgo LDFLAGS: -L${SRCDIR}/../../rust-cekirdek/target/release -lmimar_core
#include <stdlib.h>

// Mimar C ABI (tam başlık: rust-cekirdek/include/mimar.h)
void*  mimar_open(const char*);
void   mimar_close(void*);
char*  mimar_resolve(void*, const char*);
char*  mimar_recall(void*, const char*);
int    mimar_teach(void*, const char*, const char*);
void   mimar_free_string(char*);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Mimar, .mimar beynini saran küçük bir Go tipidir.
type Mimar struct{ h unsafe.Pointer }

func Open(path string) (*Mimar, error) {
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	h := C.mimar_open(cp)
	if h == nil {
		return nil, fmt.Errorf("beyin açılamadı: %s", path)
	}
	return &Mimar{h: unsafe.Pointer(h)}, nil
}

func (m *Mimar) Close() { C.mimar_close(m.h) }

// Teach: yetkili/kalıcı öğretme (soru → cevap).
func (m *Mimar) Teach(prompt, answer string) bool {
	cp, ca := C.CString(prompt), C.CString(answer)
	defer C.free(unsafe.Pointer(cp))
	defer C.free(unsafe.Pointer(ca))
	return C.mimar_teach(m.h, cp, ca) == 0
}

// Resolve: tipli direktif JSON'u (fact/rule/tool/stale/miss). Model çağırmaz.
func (m *Mimar) Resolve(q string) string {
	cq := C.CString(q)
	defer C.free(unsafe.Pointer(cq))
	r := C.mimar_resolve(m.h, cq)
	if r == nil {
		return ""
	}
	defer C.mimar_free_string(r)
	return C.GoString(r)
}

func main() {
	m, err := Open("beyin.mimar")
	if err != nil {
		fmt.Println("hata:", err)
		return
	}
	defer m.Close()

	m.Teach("iade nasıl yapılır", "14 gün içinde kargoyla iade edebilirsiniz")
	fmt.Println("resolve →", m.Resolve("ürünü nasıl geri gönderirim"))
	// Beklenen: {"tip":"fact",...,"cevap":"14 gün içinde kargoyla iade edebilirsiniz"}
}
