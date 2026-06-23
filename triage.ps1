# ============================================================
# FFXIV Character Card Studio — Screenshot Triage
# 對一整個資料夾的截圖評分，挑出值得送 VLM 的圖並裁切預覽
# 修改下方參數後直接執行: .\triage.ps1
# ============================================================

# ── 輸入 / 輸出 ─────────────────────────────────────────────
$InputDir  = "character"                 # 截圖資料夾
$OutputDir = "outputs\triage\run-001"    # 報告與裁切預覽輸出

# ── 篩選 ────────────────────────────────────────────────────
$Top         = 0           # 只保留分數最高的 N 張（0 = 全部可用的）
$MaskBackend = "rembg"     # "rembg"（任何背景）| "blue_screen"（藍色沙龍照）
$Pad         = 0.08        # 裁切框外擴比例

# ── 門檻（過濾不可用的圖；想更寬鬆就調低）──────────────────
$MinBrightness    = 40     # 主體平均亮度下限（過暗的室內/剪影會被剔除）
$MinSubjectHeight = 360    # 主體高度像素下限（太遠太小會被剔除）
$MinCoverage      = 0.012  # 主體佔畫面比例下限
$MinSharpness     = 10     # 清晰度下限（模糊/動態會被剔除）

# ============================================================
# 以下不需要修改
# ============================================================

.\.venv\Scripts\Activate.ps1
python scripts\triage_screenshots.py `
    --input-dir $InputDir `
    --output-dir $OutputDir `
    --top $Top `
    --mask-backend $MaskBackend `
    --pad $Pad `
    --min-brightness $MinBrightness `
    --min-subject-height $MinSubjectHeight `
    --min-coverage $MinCoverage `
    --min-sharpness $MinSharpness
