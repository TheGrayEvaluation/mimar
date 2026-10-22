# Homebrew formülü — macOS/Linux kurulumu:  brew install --build-from-source ./packaging/mimar.rb
# Yayından sonra: url + sha256 gerçek sürüm arşiviyle doldurulur.
class Mimar < Formula
  desc "Yapay zekanin SQLite'i — gomulu, modelden bagimsiz AI cekirdegi"
  homepage "https://github.com/TheGrayEvaluation/mimar"
  url "https://github.com/TheGrayEvaluation/mimar/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "DOLDUR_SURUM_ARSIV_SHA256"
  license "MIT"
  head "https://github.com/TheGrayEvaluation/mimar.git", branch: "main"

  depends_on "rust" => :build

  def install
    cd "rust-cekirdek" do
      system "cargo", "install", "--locked", "--root", prefix, "--path", ".", "--bin", "mimar"
    end
  end

  test do
    # argümansız çalıştırma kullanım metni basar (çıkış kodu 2) ve "Mimar" içerir
    output = shell_output("#{bin}/mimar 2>&1", 2)
    assert_match "Mimar", output
  end
end
