# source

- Source: `source.pptx`
- Total slides: 20

## Slide 1

![Slide 1 Image 1](source_readback_files/image2.png)

![Slide 1 Image 2](source_readback_files/image1.jpg)

EMOTION-AWARE INTELLIGENT LIGHT ENVIRONMENT

LumiMind

基于情绪感知的智能光环境调控系统

复旦大学「智能万象 · 光启新程」光源智能体创新设计大赛

团队成员：李彩凤，耿文俊，陈泓霖，曹君勤

指导教师：林燕丹

2026 年10月8日

## Slide 2

PART 01 · 项目背景

研究背景与问题定义

室内光环境对人体的非视觉效应——情绪、警觉性、昼夜节律——具有显著影响；

然而现有照明系统「看不见人」：缺乏对用户生理状态的实时感知与自适应调控能力。

现有系统局限

目标系统能力

静态参数 —— 色温与照度一经设定便固定

实时 ECG / EEG 生理信号感知

无生理感知 —— 无法获知用户的实时状态

多模态情绪状态评估（V/A 连续值）

→

人工调节 —— 依赖用户手动开关与调光

AI 智能推荐光照参数（色温 / 照度）

干预无验证 —— 光照调整后效果无从评估

干预效果闭环验证与持续优化

核心问题

生理信号采集 → 情绪状态评估 → 光照参数智能推荐 → 干预效果验证 的闭环自适应调控

LumiMind · 基于情绪感知的智能光环境调控系统

02 / 20

## Slide 3

![Slide 3 Image 3](source_readback_files/image3.jpg)

PART 01 · 项目背景

系统架构总览

情绪感知模型

多模态 ECG/EEG 编码器 + 跨模态融合+Qwen大模型

01

信号采集

世界模型决策

Causal Transformer 前瞻预测 + ReAct 智能体

05

02

多层记忆系统

效果评估

情绪感知

LumiMind

工作 / 情景 / 语义 / 程序四层记忆个性化

自学习闭环

RAG 知识库

35 篇光-情绪论文 + 用户生理数据双源检索

04

03

AI 问答系统

灯具调控

智能决策

LangGraph 多智能体，回答附引用与安全声明

ECG / EEG 采集 → 情绪感知 → AI 决策 → 调控 → 评估 → 回环

LumiMind · 基于情绪感知的智能光环境调控系统

03 / 20

## Slide 4

PART 02 · 生理信号采集与设备

多模态生理信号采集硬件

ECG 与 EEG 从心脏与大脑两个生理维度互补反映情绪状态

![Slide 4 Image 4](source_readback_files/image5.png)

![Slide 4 Image 5](source_readback_files/image4.jpg)

![Slide 4 Image 6](source_readback_files/image6.jpg)

建议 4:3 · ≥800×600 · 含佩戴效果

建议 4:3 · ≥800×600 · 电极帽特写

建议 4:3 · ≥800×600 · 实验环境全景

待置入真实照片

待置入真实照片

待置入真实照片

EEG 采集设备实物照片

ECG 采集设备实物照片

被试佩戴实验现场

ECG · 心脏电活动

EEG · 大脑皮层电活动

双通道互补

2 通道 · 256 Hz · 自主神经系统

14 通道 · 128 Hz · 中枢神经系统

情绪状态的完整生理刻画

LumiMind · 基于情绪感知的智能光环境调控系统

04 / 20

## Slide 5

PART 02 · 生理信号采集与设备

多模态生理信号示例

使用原始 ECG 波形、EEG 频谱监测用户的关键生理特征

![Slide 5 Image 7](source_readback_files/image8.png)

![Slide 5 Image 8](source_readback_files/image7.png)

ecg_waveform.png · 建议 9:5

spectrum.png · 建议 9:5

置入后用引线标注 R 峰位置

置入后用引线标注 α 波段

ECG 心电波形截图

EEG 脑电频谱截图

ECG 256 Hz · 12 导联

EEG 256 Hz · 32 通道

R-R 间期准周期模式

α / β / θ 频段能量分布

LumiMind · 基于情绪感知的智能光环境调控系统

05 / 20

## Slide 6

PART 02 · 生理信号采集与设备

执行端 —— 智能灯具系统

调光代码

AI 决策

Cube灯光

Cube展示

世界模型输出

Python控制cube

CCT 2000K–6500K

通过灯光控制应用或者Python代码来自动控制项目组的Cube灯光室。

灯光控制界面

![Slide 6 Image 9](source_readback_files/image10.jpeg)

![Slide 6 Image 10](source_readback_files/image9.png)

待置入真实照片

真实界面截图待置入：色温 / 照度 / 场景模式控制面板

LumiMind · 基于情绪感知的智能光环境调控系统

06 / 20

## Slide 7

PART 03 · 多模态情绪感知模型

多模态 ECG / EEG 情绪感知

将两个模型分别作为多模态编码器，融合后共同预测用户情绪。相比单模态效果更好

PhysioMamba

CBraMod

~30M

~25M

可训练参数

可训练参数

ECG 编码器

EEG 编码器

Mamba S6 选择性状态空间结构

Criss-Cross Transformer · 12 层 8 头

多模态融合

Q-Former 桥接

O(n) 线性复杂度长序列建模

空间维交叉注意力 —— 通道间关系

捕获 R-R 间期准周期模式

时间维交叉注意力 —— 长程依赖

对应自主神经系统的情绪响应

对应中枢神经系统的情绪活动

情绪状态 —— Valence / Arousal

≥82%

≥90%

5–15%

DEAP 数据集

SEED 数据集

跨模态融合

4 分类准确率

3 分类准确率

相对单模态增益

LumiMind · 基于情绪感知的智能光环境调控系统

07 / 20

## Slide 8

PART 03 · 多模态情绪感知模型 · 核心创新

轻量级跨模态桥接 —— Q-Former

问题一 · 维度鸿沟

问题二 · 序列冗余

编码器输出 d=512，LLM 嵌入空间 d=3584

576 Token 长序列直接注入将稀释语言语义

编码器输出

Cross-Attention

语义 Token

Projection Head

576 Token

32 Token

d=3584

32 个 Learnable

注入 LLM

Queries 压缩

d=512 · ECG+EEG 特征

Self-Attention 去冗余

对齐 LLM 嵌入空间

软提示前缀

选择性提取情绪语义

~10M

18:1

10×

可训练参数

Token 压缩比

训练成本降低

仅 Q-Former 模块

576 → 32

一个数量级

三阶段渐进训练

阶段一 · 编码器冻结

阶段二 · Q-Former 训练

阶段三 · LLM LoRA 微调

保留预训练生理表征

跨模态对齐学习

仅更新 <1% 参数

LumiMind · 基于情绪感知的智能光环境调控系统

08 / 20

## Slide 9

PART 03 · 多模态情绪感知模型

实时情绪状态监测

![Slide 9 Image 11](source_readback_files/image11.png)

系统主控台界面截图（main.html）

建议 1280×720 全屏截图 · 置入后按实际界面校准下方引线标注

真实截图待置入

四类情绪（正面 / 中性 / 负面 / 高唤醒）

通过监控生理状态实时给出情绪的V和A值

LumiMind · 基于情绪感知的智能光环境调控系统

09 / 20

## Slide 10

PART 04 · 世界模型与光环境决策

光照参数对情绪的神经生理学影响

光照参数

情绪与生理效应

文献依据

高色温 >4000K + 高照度

提升 Arousal，促进警觉与专注

Smolders 2014; Ru 2019

低色温 <3000K + 低照度

降低 Arousal，辅助身心放松

Mostafavi 2024

暖白光 2700–3000K

显著缓解焦虑情绪

Li X 2025

文献来源：Building and Environment · Indoor Air 等期刊；

![Slide 10 Image 12](source_readback_files/image12.jpg)

暖夜客厅 · 低色温低照度

冷白办公室 · 高色温高照度

中性书房 · 中性色温

睡前床头 · 暖光助眠

LumiMind · 基于情绪感知的智能光环境调控系统

10 / 20

## Slide 11

PART 04 · 世界模型与光环境决策

Causal Transformer 世界模型 + ReAct 智能体

学习状态转移 f(s_t, a_t) → s_t+1，前瞻预测让系统从被动响应升级为主动干预

世界模型 · Causal Transformer

ReAct 闭环 · 六节点工作流

~2.2M 参数 · 6 层 8 头 · d=256

候选方案经 Utility Score 排序与安全护栏

预测干预后的情绪变化 · 单次推理 <10ms

过滤后方可执行，观测结果回环至下一轮感知

Safety

Sense

Think

Predict

Act

Observe

安全检查

感知

思考

预测

执行

评估

CCT∈[2000,6500]K

ECG/EEG 实时状态

LLM 生成候选方案

Utility Score 排序

Python代码调节

干预前后 V/A 对比

Lux∈[50,1000] · conf>0.3

直接 END

低置信度 / 参数不安全

评估结果回环至下一轮感知

R² > 0.3

>70%

V/A 预测决定系数

Rollout 趋势一致性

世界模型拟合能力

多步预测与真实演化的一致比例

LumiMind · 基于情绪感知的智能光环境调控系统

11 / 20

## Slide 12

PART 04 · 世界模型与光环境决策

系统闭环演示·监控界面

![Slide 12 Image 13](source_readback_files/image13.png)

完整闭环演示视频

30–45 秒 · 建议置入 16:9 视频（1152×648）· 此处播放视频

感知

预测

决策

执行

评估

01

02

03

04

05

LumiMind · 基于情绪感知的智能光环境调控系统

12 / 20

## Slide 13

PART 04 · 世界模型与光环境决策

系统闭环演示·现场环境

![Slide 13 Image 14](source_readback_files/image14.png)

完整闭环演示视频

30–45 秒 · 建议置入 16:9 视频（1152×648）· 此处播放视频

感知

预测

决策

执行

评估

01

02

03

04

05

LumiMind · 基于情绪感知的智能光环境调控系统

13 / 20

## Slide 14

PART 05 · AI 智能问答系统

基于 LangGraph 的 AI 问答系统

支持光疗领域知识问答与用户个人生理数据查询双场景

知识库查询与个人生理数据

光疗领域知识日常问答

![Slide 14 Image 15](source_readback_files/image15.png)

![Slide 14 Image 16](source_readback_files/image16.png)

「现在我觉得工作非常焦虑

灯光应该怎么调？」

建议睡前 1 小时切换 2700K 暖光，

照度降至 50–100 Lux 辅助入睡。

来源：Li X 2025 · 非医疗建议

AI 问答界面截图

AI问答.png · 建议 3:2 · ≥900×600

真实截图待置入

知识库记录生理数据与ai查询知识库能力展示

包含ai对话、联网搜索能力与引用来源展示

LumiMind · 基于情绪感知的智能光环境调控系统

14 / 20

## Slide 15

PART 05 · AI 智能问答系统

论文文献库 + 用户生理数据 Embedding

来源一 · 论文文献库

35

篇本课题组

光-情绪研究论文

PDF / Word / TXT / MD 自动分块

LSH 向量检索

双源召回

融合排序

生成回答

余弦×0.7

RRF 倒数排名

附论文引用

混合评分 + RRF 重排

+关键词×0.3

融合重排

与安全声明

来源二 · 用户生理数据

历史 ECG / EEG / 情绪记录

Embedding 向量化（d=768）

时间衰减 + 语义相似度召回

个人化上下文注入

LumiMind · 基于情绪感知的智能光环境调控系统

15 / 20

## Slide 16

PART 06 · 记忆系统与个性化

跨会话记忆与个性化学习

四层记忆让系统随使用积累偏好，推荐从通用规则收敛至个性化策略

工作记忆

情景记忆

语义记忆

程序记忆

当前会话上下文

历史光干预记录

光疗领域知识

用户偏好画像

LangGraph AgentState

干预前后 V/A 对比

35 篇论文要点

从情景记忆归纳

生命周期：单次会话

生命周期：持久化

生命周期：随库更新

生命周期：定期更新

技术实现

技术实现

技术实现

技术实现

GraphState / Context

SQLite + Vector Index

RAG Knowledge Base

Preference Profile

会话内即时引用

相似情景召回

检索增强生成

推荐策略沉淀

LumiMind · 基于情绪感知的智能光环境调控系统

16 / 20

## Slide 17

PART 07 · 前端界面展示

前端交互系统

HTML + JavaScript 四页面覆盖主控台生理信号，情绪状态与光环境监控全功能

AI 问答

主控台

![Slide 17 Image 17](source_readback_files/image17.png)

![Slide 17 Image 18](source_readback_files/image18.png)

技术栈：HTML + JavaScript · 实时数据可视化 · 参数精细控制 · 零依赖轻量部署

LumiMind · 基于情绪感知的智能光环境调控系统

17 / 20

## Slide 18

PART 07 · 前端界面展示

信号采集与灯光控制可视化

生理信号实时可视化 + 光照参数精细控制 + 数据分析看板 三位一体

ECG 实时波形面板

频谱分析面板

光照控制面板

![Slide 18 Image 19](source_readback_files/image20.png)

![Slide 18 Image 20](source_readback_files/image19.png)

![Slide 18 Image 21](source_readback_files/image21.png)

ecg_waveform.png · 建议 9:5

spectrum.png · 建议 9:5

光显示界面.png · 建议 8:5

复用 P6 素材 · 标注 R 峰

复用 P6 素材 · 标注 α 波段

复用 P7 素材 · 标注 CCT/Lux 滑杆

LumiMind · 基于情绪感知的智能光环境调控系统

18 / 20

## Slide 19

PART 08 · 总结

参考文献

- [1] Smolders K C H J, de Kort Y A W, Clout M P J. Bright light and mental fatigue: Effects on alertness, vitality, performance and physiological arousal[J]. Journal of Environmental Psychology, 2014, 39: 77-91.
- [2] Ru T, Smolders K C H J, de Kort Y A W. Effects of illuminance and correlated colour temperature on daytime cognitive performance and mood[J]. Building and Environment, 2019, 163: 106322.
- [3] Mostafavi I, et al. Effects of correlated colour temperature on brain activity: A systematic review[J]. Lighting Research & Technology, 2024, 56(3): 342-367.
- [4] Li X, et al. Effects of illuminance and correlated colour temperature on emotion perception in indoor environments[J]. Indoor Air, 2025, 35(1): e13245.
- [5] Koelstra S, et al. DEAP: A database for emotion analysis using physiological signals[J]. IEEE Transactions on Affective Computing, 2012, 3(1): 18-31.
- [6] Stamatopoulos S, et al. DREAMER: A database for emotion recognition through EEG and ECG signals[J]. IEEE Transactions on Affective Computing, 2018, 9(1): 67-79.
- [7] Zheng W L, Lu B L. Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks[J]. International Journal of Automation and Computing, 2015, 12(5): 475-488.
- [8] Li J, et al. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and frozen large language models[C]. ICML, 2023.
- [9] Hu E J, et al. LoRA: Low-rank adaptation of large language models[C]. ICLR, 2022.
- [10] Hafner D, Pasukonis J, Ba J, et al. Mastering diverse control tasks through world models[J]. Nature, 2025, 640: 647-653.
- [11] Yao S, Zhao J, Yu D, et al. ReAct: Synergizing reasoning and acting in language models[C]. ICLR, 2023.
- [12] Zhou L, et al. High circadian stimulus for depression: A systematic review and meta-analysis[J]. Frontiers in Neuroscience, 2022, 16: 893044.
- [13] Zhou L, et al. Different colors on emotion: A Bayesian network meta-analysis[J]. Frontiers in Psychology, 2022, 13: 932597.
- [14] Shi Q, et al. CBraMod: Foundational models for multimodal physiological signals[C]. NeurIPS Workshop, 2025.
- [15] Fekry M, et al. Lighting impact on stress reduction: A study of correlated colour temperature[J]. Building and Environment, 2025, 270: 112497.
- [16] Chase C. LangGraph: Multi-agent systems and state graph orchestration[EB/OL]. https://github.com/langchain-ai/langgraph, 2024.
- [17] Microsoft. GraphRAG: From local to global in retrieval augmented generation[EB/OL]. https://github.com/microsoft/graphrag, 2024.
- [18] Edge P, et al. Mem0: Building production-ready AI agent with scalable long term memory[EB/OL]. https://github.com/mem0ai/mem0, 2024.
- [19] Hochreiter S, Schmidhuber J. Long short-term memory[J]. Neural Computation, 1997, 9(8): 1735-1780.

19 篇参考文献支撑 · 覆盖光-情绪效应、生理信号建模与 LLM 智能体前沿

LumiMind · 基于情绪感知的智能光环境调控系统

19 / 20

## Slide 20

![Slide 20 Image 22](source_readback_files/image22.jpg)

PART 08 · 总结

未来工作与致谢

扩大被试规模

01

目标 100+ 人 · 增强模型跨人群泛化能力

扩展生理模态

02

皮肤电 EDA · 呼吸频率等更多信号源

场景拓展

03

从办公场景走向居家光疗愈应用

致谢

复旦大学光源智能体创新设计大赛 · 欧普照明 · 林燕丹老师指导

- 团队联系方式：ydlin@fudan.edu.cn
- GitHub 仓库：https://github.com/KrGage/lumimind.git

感谢各位老师聆听

LumiMind · 20 / 20 · Thank You
