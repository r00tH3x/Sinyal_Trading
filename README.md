# Sinyal_Trading

## 📄 Deskripsi

Sinyal_Trading adalah sebuah skrip Python sederhana yang berfungsi
sebagai bot sinyal trading. Bot ini dapat membantu menghasilkan sinyal
beli/jual berdasarkan logika strategi yang tertanam di dalam `bot.py`.
Script ini cocok untuk edukasi, riset, dan eksperimen dalam
mengembangkan sistem sinyal trading otomatis.

> ⚠️ **Catatan:** Tool ini bukan alat untuk menjamin profit. Segala
> bentuk risiko trading sepenuhnya menjadi tanggung jawab pengguna.

------------------------------------------------------------------------

## 🔧 Fitur Utama

-   Script tunggal `bot.py` yang sangat ringan.\
-   Mudah dimodifikasi untuk menambah strategi trading.\
-   Cocok dijalankan di PC, server, VPS, Termux, maupun Linux
    environment.\
-   Dapat diperluas dengan API exchange, indikator teknikal, sistem
    alert, dan banyak fitur lainnya.

------------------------------------------------------------------------

## 🛠️ Prasyarat

-   Python 3.x\
-   Internet (jika bot nanti terhubung ke API)\
-   (Opsional) Virtual environment

------------------------------------------------------------------------

## 🚀 Cara Install & Menjalankan

``` bash
# 1. Clone repository
git clone https://github.com/r00tH3x/Sinyal_Trading.git

# 2. Masuk folder
cd Sinyal_Trading

# 3. (Opsional) Buat virtual environment
python -m venv venv

# Aktifkan:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Jika ada requirements.txt
pip install -r requirements.txt

# 5. Jalankan bot
python bot.py
```

------------------------------------------------------------------------

## 📁 Struktur Proyek

    Sinyal_Trading/
     ├── bot.py       # Script utama
     ├── README.md    # Dokumentasi

------------------------------------------------------------------------

## 📘 Cara Modifikasi

-   Buka `bot.py`
-   Ubah logika strategi agar sesuai kebutuhan trading kamu\
-   Tambahkan:
    -   Indikator teknikal (RSI, MACD, MA, Bollinger Bands, dll)
    -   API exchange seperti Binance, Bybit, atau OKX
    -   Notifikasi Telegram\
    -   Sistem log file\
    -   Mode paper trading vs. live trading

------------------------------------------------------------------------

## ⚠️ Disclaimer

-   Script ini tidak menjamin keuntungan\
-   Gunakan hanya untuk edukasi\
-   Jika kamu menambahkan eksekusi order otomatis, pahami seluruh
    risikonya\
-   Developer tidak bertanggung jawab atas kerugian apa pun

------------------------------------------------------------------------

## 💡 Rencana Pengembangan 

-   Integrasi ccxt untuk live chart data\
-   Auto trading / semi-auto trading\
-   Integrasi technical indicator\
-   Mode backtesting\
-   Mode monitoring 24/7 (daemon / systemd)

------------------------------------------------------------------------

## ✨ Penutup

Gunakan tool ini untuk belajar membangun sistem trading otomatis.
Modifikasi sesuai gaya trading kamu agar bot lebih optimal.\
Semoga bermanfaat!
