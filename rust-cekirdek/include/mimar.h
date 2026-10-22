/* Mimar — C ABI başlığı. Gömülü AI çekirdeği (.mimar beyin).
 * Linkle: mimar_core.dll (Win) / libmimar_core.so (Linux) / .dylib (macOS)
 * Felsefe: Mimar YÖNLENDİRİR, host ÇALIŞTIRIR (FFI model çağırmaz).
 */
#ifndef MIMAR_H
#define MIMAR_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct MimarBrain MimarBrain;
MimarBrain* mimar_open(const char* path);   /* yoksa oluşturur; ""=bellek; hata=NULL */
void        mimar_close(MimarBrain* b);
char*       mimar_resolve(MimarBrain* b, const char* query);  /* direktif JSON; MODEL ÇAĞIRMAZ */
char*       mimar_recall(MimarBrain* b, const char* query);   /* cevap ya da NULL */
int         mimar_teach(MimarBrain* b, const char* prompt, const char* answer); /* 0=ok */
int         mimar_feedback(MimarBrain* b, const char* query, int good);         /* good=1/0 */
char*       mimar_stats(MimarBrain* b);
char*       mimar_version(void);

/* Çok-kiplilik (KİP-AGNOSTİK): host medyayı parmaklar → vektör → teach/resolve.
 * embed_image/audio: çıktı out_vec'e yazılır (en az 256 double), dönüş = uzunluk (0=hata). */
int         mimar_embed_image(const unsigned char* gray, size_t w, size_t h,
                              double* out_vec, size_t out_cap);
int         mimar_embed_audio(const double* samples, size_t n,
                              double* out_vec, size_t out_cap);
int         mimar_teach_vec(MimarBrain* b, const char* key,
                            const double* vec, size_t n, const char* answer);  /* 0=ok */
char*       mimar_resolve_vec(MimarBrain* b, const double* vec, size_t n);     /* direktif JSON */

/* Güvenlik: prompt-injection/jailbreak risk etiketleri (JSON dizi; []=temiz). */
char*       mimar_scan(const char* text);

void        mimar_free_string(char* s);     /* dize döndürenlerin sonucunu serbest bırak */
#ifdef __cplusplus
}
#endif
#endif
