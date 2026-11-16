# bravegpuwinner：FlagOS S2 赛道2候选复现说明

本候选把 MiniCPM5-2B 的分页注意力、缓存放置和部分线性层量化接在现有 FlagOS 分派路径中。分页注意力重用同组查询与历史数据；缓存放置争取让连续逻辑页落在连续物理页；量化路径把部分权重与激活转成 INT8，再恢复 BF16 输出。本文说明实现归属、固定版本、安装及复用现有测试和官方吞吐命令的方法。2026-11-16 首次仅完成本机静态整理、未推送；随后用户授权将本分支共享到自有 fork，供组员完成报告与源码审阅。整理树尚未真机测试，官方社区 PR 和赛事正式提交仍为 `blocked_pending_target_runtime`。

## 1. 固定源码与平台条件

队名为 **bravegpuwinner**。plugin 固定官方比赛分支起点 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`，FlagGems 固定官方 v5.3.5 起点 `a7620cc191a0b42e040194622c5758b22a7a25dc`。上游 vLLM 只读参考为 v0.24.0 / `ee0da84ab9e04ac7610e28580af62c365e898389`；镜像中的 vendor vLLM 二进制版本与实际 import 路径须在目标机重新记录，不能用只读 checkout 证明安装包身份。

候选分支均为 `bravegpuwinner/s2-community-candidate`，本 plugin 必须配套 **FlagGems `10d19b55c55cac76ce6573078c4efd0efd39ef3a`**。不允许只安装公开发行版 FlagGems。保留的 BF16 参考分支均为 `bravegpuwinner/s2-bf16-reference`：FlagGems `5276bb367`，plugin `283a593`。两对分支独立保留，切换对照时必须成对切换、重启服务并核实实际导入路径。本候选自身提交由所在 checkout 的 `git rev-parse HEAD` 和提交外的 release index 标识，避免把自己的 SHA 写回提交而产生循环。

| 平台 | 适用边界与已知环境 |
| --- | --- |
| Iluvatar BI-V150 | 历史 BF16 为 32 GiB、IX-ML 4.4.0、driver/CoreX 4.5.0、Python 3.12.11、Torch 2.10.0+corex.4.5.0、FlagTree 0.6.2a3+iluvatar3.6 / Triton 3.6；本量化候选依赖 ixformer 0.7.0+corex.4.5.0。目标镜像仍需重新核实。 |
| MetaX C500 | 保留已验 MetaX 实现的 `1bbd3c585` / `47882ba` 历史；既有租赁预验证环境为 C500 64 GiB、driver 3.8.30、MACA 3.7.0.38、Python 3.12.11、Torch 2.8.0+metax3.7.0.7、Triton 3.0.0+metax3.7.0.7。整理树尚未回归，租赁记录不代表组委会官方资格。 |

使用各平台匹配的官方镜像，不在两平台间混装数值库。当前没有租赁中的目标服务器，本机禁止安装、导入或运行项目。

## 2. 核心机制与代码映射

“Prefill”处理一批新输入，“Decode”每步生成后续 token；GQA 让多个查询头共用较少的 KV 头。本模型为 16 查询头、2 KV 头、head dimension 128。专用分支只接管符合资格的请求，未知 dtype、布局或功能继续走原有实现。

| 机制 | FlagGems 实现 | plugin 接入 |
| --- | --- | --- |
| 分页 GQA、Q32 查询复用、KV128 历史块、运行时 token 数、混合批次与 Decode 分段 | `src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa_prefill.py`、`paged_gqa_decode.py` | `vllm_fl/dispatch/backends/vendor/iluvatar/paged_gqa_prefill.py` 的选择、兼容性和回退分支；对应两仓同名测试 |
| BF16 GEMM 访问次序与兼容形状路径 | `src/flag_gems/runtime/backend/_iluvatar/ops/mm.py` | 继续使用原有 FlagGems 注册分派 |
| BF16 K/V slabs 与 extent-first 放置 | 上述 attention 接口读取真实 stride；KV 数值仍为 BF16 | `IluvatarPagedGQAPrefillBackend.get_kv_cache_stride_order_for_spec`、`vllm_fl/kv_cache/extent_first_full_attention.py`、`platform.py` 及 runner 两处 KV 接口 |
| SmoothQuant W8A8 | `src/flag_gems/runtime/backend/_iluvatar/ops/smooth_w8a8.py` 的融合 inverse-s/A8/RNE、加载时 W8/scale、同 ABI Triton 回退 | `vllm_fl/quantization/w8a8/iluvatar_smooth.py`、`iluvatar_smooth_config.py` 与 recipe；由 `platform.py` 调用 `maybe_enable` |
| 保留 MetaX 路径 | 已验排序索引实现 | 已验 attention 累积长度复用；不接入 BI 量化与 KV spec |

extent-first 只在精确的 42 层、page16、BF16 KV、单 rank GQA 合同成立时转换 spec。容量按四页取整并受模型最大长度约束，每请求最多多持有三页；在 free queue 前八个 anchor 中找完整的连续四页，失败时使用原池分配。refcount、hash、回收和共享所有权仍通过原池接口维护。改变 cached-free 页的选择可能改变后续缓存命中和准入状态，这些影响必须完整计入端到端测试。

整理时从冻结归档只复制五个 FlagGems 文件、八个 plugin 文件；runner 从 BF16 起点仅移植 stride-spec 分派与 iluvatar spec 转换两处。没有携带归档中的 MetaX downpack 加载、重载或稀疏更新 hook，因为其依赖的 `mm_tile_packed` 不在冻结 FlagGems 中；没有携带当前开发目录的 W8 原型、geometry、packed 或 top-p 实验。归档缺少部分非运行文件并含缓存，不能当作完整仓库覆盖或做删除式同步。

## 3. 量化激活、recipe 与实现归属

量化不是 BF16 等价变换。`PlatformFL.check_and_update_config` 在 iluvatar 平台调用 `maybe_enable`；符合 MiniCPM5-2B 稠密 BF16、无 bias、无 LoRA/推测解码、单 rank 及兼容 cache 等条件后，代码会自动设置 `model.quantization="bi_smooth_native_w8a8_v2"`、版本 2 的量化配置和 `vllm_config.quant_config`。没有新增 CLI 量化开关并不等于得到官方量化资格批准，最终提交仍须按当前规则确认此路径。

recipe 是随源码附带的缩放及校准元数据，大小 1,435,459 字节，不含完整模型权重。历史来源为独立 WikiText-2 raw train 32 段、validation 8 段，每段 768 token，固定 alpha=0.5；数据源 `Salesforce/wikitext` revision `b08601e04326c79dfdd32d625aee71d232d685c3`。校准不使用 MATH 或官方 benchmark；validation 参与选层，不能称其为独立最终质量测试。84 个 QKV/gate-up 角色中历史启用 83 个，其余线性层、embedding、lm_head 与 KV 保持 BF16。加载时逐个检查原始权重摘要、形状、缩放合法性，转换为 raw INT8 `[N,K]` 与 FP32 scale，避免保留 BF16 权重镜像。

在 `K=2048,N=2560/12288` 的实际角色上，GEMM 调用已有 `ixformer.inference.functions.w8a8.w8a8(..., format="TN")`，厂商 INT8 GEMM 明确归属 ixformer。自研工作是 inverse-s 与逐 token A8 的融合、RNE 舍入与边界填充、加载时 W8/scale 转换、recipe/模型哈希 ABI 和受限框架接入；不把 SmoothQuant 算法或已有厂商 GEMM 称作自研。其他已支持形状保留相同量化 ABI 的 Triton 路径，厂商错误不被捕获后偷偷替换实现。

`pyproject.toml` 新增 recipe 的 setuptools package-data 声明，防止非 editable 安装漏带文件。本次仅用 ZIP 目录和 pickle opcode 静态扫描，没有 `torch.load`、项目导入或反序列化；wheel 尚未构建或安装验证。源码、recipe 和历史厂商 wrapper/binary 摘要记录在 [`submission/manifest.json`](../../submission/manifest.json)。

## 4. 目标机安装与现有验证命令

以下仅为目标机复现命令，本次未执行。先取得当前官方认可的平台、镜像和模型；在两个仓库中核对固定提交，保留现有 vendor vLLM/驱动/数值库，然后 editable 安装。`/workspace` 是本说明的目标机工作目录约定。

```bash
cd /workspace/FlagGems
git checkout 10d19b55c55cac76ce6573078c4efd0efd39ef3a
pip install --no-build-isolation -e .
cd /workspace/vllm-plugin-FL
git checkout bravegpuwinner/s2-community-candidate
git rev-parse HEAD
pip install --no-build-isolation -e .
```

目标机必须记录两仓完整提交、dirty 状态、真实 import 路径与 recipe 文件是否随安装存在。BI 还需确认 83 个实际量化模块、量化 ABI/厂商版本及 extent manager 激活，C500 需确认不走 BI 量化与 KV 路径；源码中有符号不代表服务已激活。非 editable 的 wheel 包含 recipe 与实际加载位置也要另行验证，本次没有构建结果。

在 BI 上先复用两仓已有回归测试，覆盖 GQA、动态长度、graph、布局和分派回退：

```bash
cd /workspace/FlagGems
python -m pytest -q tests/test_iluvatar_paged_gqa_prefill.py
cd /workspace/vllm-plugin-FL
python tests/unit_tests/dispatch/test_iluvatar_paged_gqa_prefill.py
```

这两条命令不代表 W8 与 extent 已全覆盖。继续复用既有 W8 算术、量化加载、extent 所有权/低容量/取消/抢占测试以及 short/long/mixed smoke，将原夹具明确指向这对整理后的固定提交；旧脚本若仍绑定历史快照，不能直接执行后宣称测试了本候选。每项结果均须记录实际加载源码，任一数值、所有权、资源或请求失败立即停止长测和发布，不新建另一套测试平台。

正确性通过后，使用原服务参数；以下是历史 BI 命令。其他平台的参数遵循当前官方要求，并在 BF16/候选两臂保持一致。

```bash
cd /workspace/vllm-plugin-FL
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B --port 9031 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

在独立终端执行未改的官方 wrapper；4K/1K 为 64 并发、256 请求，16K/1K 为 64 并发、128 请求。保持原 sampling、ignore-eos、生成长度和缓存设置，不增加 observer、额外 warmup 或预请求。

```bash
cd /workspace/vllm-plugin-FL
python benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

BI 应在同环境按成对 BF16/候选各完整执行两景四轮，保留全部慢轮，仅依原 `RUNS=4,SKIP_FIRST=1` 统计后三轮。核对每轮成功请求、输入输出 token、退出状态、TTFT、吞吐、波动和稳定噪声门；BI Mean TTFT 历史限制为 11689.27/605254.65 ms，若当前官方规则有更新，以新规则为准。C500 需先过既有 operator/service 回归，再以同样完整双景四轮检查跨平台退化。任一正确性、资源、失败请求或 TTFT 不通过即停止后续长测与发布，不删除慢轮。

## 5. 历史结果与发布前条件

下面仅报告 2026-10-29 冻结实验 `bi150-20261029-051419Z-bounded-extent-first` 的稳定吞吐，无法证明 11-16 整理树可复现。两臂各场景四轮，固定弃首轮，统计后三轮样本均值和标准差；每臂共 1536 个成功请求，输入 12,582,912、输出 1,572,864 token，wrapper 均正常退出。原数值摘要完整保存在 [`submission/historical-results.json`](../../submission/historical-results.json)，内部原始 final-summary、日志和失败历史保持原件，不随社区仓库公开。

| BI 历史场景 | 同机 BF16 tok/s | 冻结候选 tok/s | 提升 | 候选 CV | Mean TTFT：BF16→候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K/1K | 3930.61 ± 55.94 | 5034.83 ± 31.43 | 28.09% | 0.624% | 4165.32→3203.80 ms |
| 16K/1K | 3162.79 ± 1.54 | 4293.03 ± 5.28 | 35.74% | 0.123% | 157162.98→115257.03 ms |

相对组委会公布基线的 2.48/4.69 倍仅为吞吐比值，不是官方综合分。此前 BF16 的 3864.10/3164.70 tok/s 与 102/105 是另一源码身份；此前 MetaX 的 7795.68/8551.02 tok/s 与 102/105 属历史租赁预验证。上述精度都不能继承给量化候选，两个平台的新整理树也尚未运行。

用户最新要求先证明正确性与稳定复现后提交，目前没有目标服务器，并明确暂不运行 105 题。因此 `submitted_tree_runtime_verified=false`、`submitted_tree_stable_throughput_verified=false`、`model_accuracy_verified=false`、`full105_run=false`。首次整理时没有 push；当前分支按用户授权共享到自有 fork 供团队报告审阅，没有创建官方社区 PR，也未提交正式赛事 ZIP。后续先完成本文正确性和双平台吞吐复现；完整 MATH-500 Level 3 仍须单独获得用户授权并通过，方能满足最终竞赛质量交付门。

官方依据：[当前赛题](https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn)、[plugin 比赛分支](https://github.com/flagos-ai/vllm-plugin-FL/tree/flagos-2026-s2)、[FlagGems v5.3.5](https://github.com/flagos-ai/FlagGems/tree/v5.3.5)。本候选保留原主体选择逻辑、benchmark 和服务 workload；量化资格、组委会认可的平台与最终提交文件仍以当前官方要求为准。
