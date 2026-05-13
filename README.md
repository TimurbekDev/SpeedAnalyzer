# SpeedAnalyzer

Tezkor va oddiy foydalanish uchun README. Bu loyiha kameradan yoki videodan avtomobil aniqlash va tezlik hisoblash uchun yaratilgan.

## Tez boshlash (Windows)

1. Virtual muhitni faollashtiring (PowerShell):

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned)
& .\.venv\Scripts\Activate.ps1
```

Yoki cmd/bash uchun:

```bash
.\.venv\Scripts\activate
# yoki
source .venv/bin/activate
```

2. Dependensiyalarni o'rnating:

```bash
pip install -r requirements.txt

```

## Speed analyzer ishga tushirish

Loyiha ildizida mavjud skript bilan ishlang (`speed_analyzer_pro.py` misol):

```bash
py speed_analyzer_pro.py 
```
