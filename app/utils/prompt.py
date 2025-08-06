def build_prompt(text: str, context: str) -> str:
    return f"""
Berikut adalah referensi tata bahasa Indonesia yang benar:

{context}

Kalimat asli:
{text}

Perbaiki kalimat di atas dengan ketentuan:
- Perbaiki EJAAN yang salah (contoh: "sya" harus jadi "saya").
- Gunakan struktur tata bahasa Indonesia yang benar.
- Gunakan kata ganti orang yang konsisten.
- Jangan menambahkan kata baru di luar kalimat.
- Jangan ganti makna kalimat.
- Tulis HANYA hasil kalimat final — TANPA penjelasan, TANPA pembuka, TANPA tanda kutip.

Kalimat final:
"""
