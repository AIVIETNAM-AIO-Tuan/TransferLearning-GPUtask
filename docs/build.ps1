# Dich bao cao ra PDF.
#
#   .\build.ps1                      -> dich BAO_CAO_TIEN_DO.tex
#   .\build.ps1 -Tex TEN_KHAC.tex    -> dich file khac
#
# VI SAO KHONG DUNG latexmk: `latexmk` la script PERL. May nay khong cai Perl nen
# MiKTeX bao "could not find the script engine 'perl'". Do KHONG phai loi file .tex.
# Goi thang `xelatex` la xong.
#
# VI SAO CHAY 3 LAN: luot 1 sinh .aux/.toc, luot 2 dung so trang cho muc luc va do
# rong longtable, luot 3 chot lai khi do rong vua doi. It hon 3 luot thi muc luc
# hoac bang dai co the sai.
#
# "critical issue: You are running MiKTeX on an unsupported version of Windows" chi
# la loi than phien — MiKTeX 24.1 khong nhan ra Windows 11 build 26200. PDF VAN RA
# dung. Vi vay script kiem PDF va so loi trong .log, KHONG kiem ma thoat cua xelatex.

param([string]$Tex = 'BAO_CAO_TIEN_DO.tex')

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot

if (-not (Test-Path $Tex)) { Write-Host "Khong thay $Tex" -ForegroundColor Red; exit 1 }
$base = [System.IO.Path]::GetFileNameWithoutExtension($Tex)
$pdf  = "$base.pdf"
$log  = "$base.log"

# Mot so shell co muc PATH hong (vd 'WindowsApps\python.exe\' bi coi la thu muc) lam
# MiKTeX chet ngay khi khoi dong. Loc ra cho chac.
$clean = ($env:PATH -split ';' | Where-Object { $_ -and ($_ -notmatch 'WindowsApps') }) -join ';'
$saved = $env:PATH
$env:PATH = $clean

$truoc = if (Test-Path $pdf) { (Get-Item $pdf).LastWriteTime } else { [datetime]::MinValue }

try {
    foreach ($i in 1..3) {
        Write-Host "luot $i/3 ..." -NoNewline
        xelatex -interaction=nonstopmode $Tex *> $null
        Write-Host " xong"
    }
} finally {
    $env:PATH = $saved
}

# --- Nghiem thu: PDF co MOI khong, va .log con dong loi nao khong ---
if (-not (Test-Path $pdf)) { Write-Host "`nTHAT BAI: khong sinh duoc $pdf" -ForegroundColor Red; exit 1 }
if ((Get-Item $pdf).LastWriteTime -le $truoc) {
    Write-Host "`nTHAT BAI: $pdf khong duoc ghi lai (dich vo o luot cuoi)" -ForegroundColor Red; exit 1
}

$loi = @(Select-String -Path $log -Pattern '^! ' -Encoding utf8)
$trang = (Select-String -Path $log -Pattern 'Output written on .*\((\d+) page' -Encoding utf8 |
          Select-Object -Last 1).Matches.Groups[1].Value

if ($loi.Count -gt 0) {
    Write-Host "`n$pdf co ra nhung .log con $($loi.Count) loi:" -ForegroundColor Yellow
    $loi | Select-Object -First 5 | ForEach-Object { Write-Host "  $($_.Line)" }
    Write-Host "  (xem day du trong $log)"
    exit 1
}

Write-Host "`nOK  ->  $pdf  ($trang trang, 0 loi)" -ForegroundColor Green

# Don file phu tro; giu .log de con truy loi khi can.
Remove-Item -Path "$base.aux","$base.toc","$base.out" -ErrorAction SilentlyContinue
