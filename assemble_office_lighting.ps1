Add-Type -AssemblyName System.Drawing

$source = 'C:\Users\28646\AppData\Local\Temp\codex-clipboard-32516661-4e62-4fbd-b89f-cdc4bdfb5baf.png'
$bright = 'C:\Users\28646\.codex\generated_images\01a18aca-f4d0-7f03-8959-257d99c3cc88\exec-23db58bd-eb9c-4dac-b5f2-8cbc47ac4f0a.png'
$warm = 'C:\Users\28646\.codex\generated_images\01a18aca-f4d0-7f03-8959-257d99c3cc88\exec-3299a557-2cbf-473e-8ba7-89774365bc82.png'
$output = Join-Path $PSScriptRoot '办公区_四种光环境对比.png'

$panelWidth = 430
$panelHeight = 305
$gap = 8
$margin = 8
$canvas = New-Object System.Drawing.Bitmap ((4 * $panelWidth) + (3 * $gap) + (2 * $margin)), ($panelHeight + (2 * $margin))
$graphics = [System.Drawing.Graphics]::FromImage($canvas)
$graphics.Clear([System.Drawing.Color]::FromArgb(17, 23, 36))
$graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality

$images = @(
    [System.Drawing.Image]::FromFile($source),
    [System.Drawing.Image]::FromFile($bright),
    [System.Drawing.Image]::FromFile($warm)
)
try {
    $destinations = @(0, 1, 2, 3) | ForEach-Object {
        New-Object System.Drawing.Rectangle (($margin + $_ * ($panelWidth + $gap))), $margin, $panelWidth, $panelHeight
    }
    $graphics.DrawImage($images[0], $destinations[0], (New-Object System.Drawing.Rectangle 10, 3, 430, 305), [System.Drawing.GraphicsUnit]::Pixel)
    $graphics.DrawImage($images[0], $destinations[1], (New-Object System.Drawing.Rectangle 459, 3, 430, 305), [System.Drawing.GraphicsUnit]::Pixel)
    $graphics.DrawImage($images[1], $destinations[2])
    $graphics.DrawImage($images[2], $destinations[3])
    $canvas.Save($output, [System.Drawing.Imaging.ImageFormat]::Png)
    Write-Output $output
} finally {
    $graphics.Dispose()
    $canvas.Dispose()
    $images | ForEach-Object { $_.Dispose() }
}
