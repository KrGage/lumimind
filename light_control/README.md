# 单灯手动控制

主界面 `/page/home` 的“环境光效实时预览”展开后，可输入目标照度，并用色温滑块设定目标色温。只有点击“下发到灯具”才调用 `POST /api/light/manual`；亮度滑块与页面光学指标是动画示意，不控制灯具，也不是实测值。

首次现场使用前，将 `light_config.json` 的 `lamp_ip` 从 `null` 改为实际灯具的 IPv4 地址，核对 UDP 端口、`room_id` 与 `runtime/Calibrations/Room1_Calibration.json` 是否适用于当前单灯和测量位置。不要依据示例地址推断现场地址。

电脑需运行 Windows x64、Python 3.10+、.NET Framework 4.7.2+；本目录带 `LightBridge.exe`、DLL 和标定文件，无需另编译。后端从 `ai_agent/backend` 启动时应只运行一个 worker，且通过本机页面访问手动灯控。

`calculate_recipe(lux, cct)` 仅用 SDK 求解五通道 PWM，不发送网络报文。`set_light(lux, cct)` 求解后向灯具发送一次 UDP。返回 `status=sent` 只表示电脑完成发送；`hardware_confirmed=false`，需用 ULTRA 的新鲜读数核对实际照度与色温。本手动接口目前不自动纠偏。

后端用 `manual_light_commands` 表记录命令编号、目标和执行状态。相同命令编号的成功请求不会再次发送；失败或发送结果未知时，也不会自动重发。若状态未知，先核对灯具与测量数据。
