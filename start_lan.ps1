param(
    [int]$Port = 8000
)

$agentRoot = $PSScriptRoot
$projectRoot = Split-Path -Parent $agentRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { 'python' }

Write-Host 'LumiMind 局域网服务即将启动。'
Write-Host "本机访问: http://127.0.0.1:$Port/"
$lanAddresses = [System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() |
    Where-Object { $_.OperationalStatus -eq 'Up' -and $_.NetworkInterfaceType -in @('Wireless80211', 'Ethernet') } |
    ForEach-Object {
        $properties = $_.GetIPProperties()
        if ($properties.GatewayAddresses.Count -gt 0) {
            $properties.UnicastAddresses |
                Where-Object { $_.Address.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork } |
                ForEach-Object { $_.Address.IPAddressToString }
        }
    }
if ($lanAddresses) {
    $lanAddresses | ForEach-Object { Write-Host "其他设备: http://${_}:$Port/" }
} else {
    Write-Host "其他设备: http://<此电脑的 WLAN/以太网 IPv4 地址>:$Port/"
    Write-Host '用 ipconfig 查看本机 IPv4 地址。'
}
Write-Host '按 Ctrl+C 停止服务。'

Push-Location (Join-Path $agentRoot 'backend')
try {
    & $python -m uvicorn app.main:app --host 0.0.0.0 --port $Port
} finally {
    Pop-Location
}
