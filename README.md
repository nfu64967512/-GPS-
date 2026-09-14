# Indoor Trajectory Studio — 室內 3D 軌跡規劃

用 **動作捕捉 (OptiTrack)** 做室內非 GPS 定位，讓 **ArduPilot Copter** 多軸無人機在
**5×5×3 m** 捕捉空間內飛出 5 種指定軌跡 (+ 隨機手飛)，並在飛行中自由變化高度，輸出可餵給
**VIO / 預測模型** 的測試資料。

軌跡輸出格式對齊 [aeroplan-studio](https://github.com/nfu64967512/aeroplan-studio)
的 **QGC WPL 110 `.waypoints`**（與 Mission Planner / QGroundControl 相容）。

| 軌跡 | 說明 |
|---|---|
| Reciprocate 往返 | 沿 X 來回 (去/回兩條平行線) |
| Rectangle 矩形 | 矩形周界 |
| Figure eight 8 字 | 水平 8 字 (Gerono lemniscate)，原點交叉 |
| Circle 圓形 | 等半徑圓 |
| Zigzag 鋸齒 | 沿 X 前進、Y 上下對角，折返閉合 |
| Random walk 隨機手飛 | 隨機轉折點 + 閉合平滑樣條，模擬人手在房間裡隨意繞飛；**整段就是一條不重複的路徑**（長度自動配合目標工時、圈數 1、不用 DO_JUMP），同 seed 同路徑、換 seed 換一條（`patterns.random_walk`：`seed`、`n_points`（auto / 整數）、`turn_sigma_deg` 轉向幅度、`turn_max_deg` 轉向上限；GUI 有對應群組）。配 `altitude.mode: smooth_random` 就是全隨機 3D 手飛 |

所有軌跡都疊加 **垂直高度調變**（sine / 三角波 / 平滑亂數 / 階梯）成為真 3D；圈數自動調整，
讓 **純飛行工時 ≥ 200 s（目標 240~300 s）**——預設以 **AUTO 任務的航線時間**為準
（`flight.duration_basis: auto`，只飛 AUTO 的工作流程；`guided` 則沿用 GUIDED 串流軌跡時間）。

- 高度是**沿路徑弧長的函數** z = f(s/S)（`altitude.cycles` = 全程起伏次數）。`.waypoints` 預設用
  DO_JUMP 重複一圈時，起伏會自動調成每圈整數次（見「輸出格式」）；`repeat: unroll` 時則把
  次數調成與圈數互質，讓各圈在 3D 中相位錯開、不重疊。
- **階梯狀升降 (`altitude.mode: stair`)**：平飛保持高度 → 以固定垂直速度升/降一階 → 再平飛，
  逐階升到 `base+amplitude` 再逐階降到 `base−amplitude`（一次起伏 = 一個 `cycles`）。
  `steps` 決定由最低到最高分幾階（每階高 = 2·amplitude/steps；1 = 方波），
  `ramp_speed` 是階間升降的垂直速度（`auto` = 80% × min(`flight.speed_up`/`speed_down`,
  √(`max_speed`²−`cruise_speed`²))，上升/下降各自取，保證既不觸發垂直速度警告、升降時的
  3D 合成速度也不超過 `max_speed`；也可給數值）。過渡的路徑寬度 = 巡航速度 × 一階高 / 升降速度
  （巡航時垂直速度恰為設定值），置中於理想換階位置，起點/終點都在 `base`；若 cycles 太多、
  路徑太短使過渡 ≥ 每階寬度，平台會消失、退化成三角波並由安全檢查警告。

## 工時預估模型（動態速度剖面）

工時不是「弧長 ÷ 定速」——室內多旋翼會在起步/收尾加減速、在硬轉角減速
（90° 矩形角、鋸齒、180° 折返幾乎要停下），曲線受向心加速度限制，且
ArduPilot 4.x 是 jerk 受限的 S 曲線。`core/speed_profile.py` 對折線做
**時間最佳化參數化**：加密 → 逐點速度上限（曲率 + 轉角）→ 前向/後向掃描
（加減速可達性）→ 梯形積分 + jerk 修正。

- **轉角過彎速度由「5 cm 級接受半徑」決定**：容許切角半徑
  `R = r·cos(δ/2)/(1−cos(δ/2))`（δ=轉向角, r=`waypoints.accept_radius`，
  對應 `WPNAV_RADIUS=5`），過彎速度 `v = sqrt(lat_accel·R)`；180° 折返 R→0
  視為近懸停再出發。δ 一律以**水平 (XY) 幾何**計算——飛控的轉角邏輯是水平面的，
  AUTO 估時餵的 3D 航點折線即使高度在變，180° 折返仍是 180°（垂直方向另由
  `speed_up/down` 限制）。
- **jerk 主導的有效加速度**：室內 0.5 m/s、`jerk=1` 下，一次 0.5 m/s 的變速
  達不到加速度上限（由 jerk 主導），故時間以有效加速度
  `a_eff = min(accel, ½·sqrt(jerk·cruise))` 計；`accel` 高於此門檻時對工時
  幾乎無影響（符合實機），低於門檻才會拖慢——預估工時對 `accel` 單調不遞增。
- **GUIDED 串流的軌跡本身就依此剖面取時**，因此「預估工時 == 串流時長 ==
  實飛時長」（追蹤正常時）；轉角處 yaw 設點另做角速度限制（`yaw.max_rate`）。
- **AUTO 任務另行預估**：對「實際匯出的航點折線（3D）」跑同一套剖面，再加起飛/
  降落，即上傳 `.waypoints` 切 AUTO 的整段任務時間（GUI/CLI 都會顯示）。3D 折線上
  `cruise_speed` 與過彎速度都是**水平**速度（WPNAV_SPEED），斜段的 3D 速度 = 水平速度 /
  cos(傾角)，垂直分量另受 `speed_up/down` 限制——與 GUIDED 軌跡的合成速度一致。
- **工時基準 `flight.duration_basis`**：`auto`（預設）以 **AUTO 航線時間**（compact 版航點
  折線、含進場、不含起飛/降落）決定自動圈數並判定 200~300 s；AUTO 通常比 GUIDED 長
  5~25 s（進場、高度斜段、精確轉角），只飛 AUTO 時用這個才不會超時。`guided` 為舊行為
  （以 GUIDED 軌跡時間為準）。GUI 統計欄第一行顯示的就是基準工時，另一種退為參考。
- 動力學參數在 `config/default.yaml` 的 `flight` 區
  （`accel`=WPNAV_ACCEL、`lat_accel`、`jerk`=WPNAV_JERK、
  `speed_up/down`=WPNAV_SPEED_UP/DN、`corner_min_speed`）。**預設是刻意調低的
  室內保守值，皆遠低於原廠**（原廠 WPNAV_ACCEL=250 cm/s²、WPNAV_JERK=5、
  WPNAV_SPEED_UP=250 cm/s²），請依實機 WPNAV_* 校準。
  > **轉角加速度**：實機 AUTO 的轉角加速度是 `WPNAV_ACCEL_C`，內定 0 表示
  > **2×WPNAV_ACCEL**。本專案 `lat_accel` 預設取保守的 1×`accel`（過彎較慢、
  > AUTO 工時偏長是安全側；GUIDED 串流也較溫和）。若要讓 AUTO 預估更貼近實機、
  > 願意接受較快的過彎，設 `lat_accel: 2.0`（=2×accel）——此舉主要影響轉角密集
  > 的軌跡（如 zigzag），平滑曲線幾乎不變。
- `speed_profile: constant` 可切回舊版定速（對照用）。

以預設參數為例，舊定速模型的低估幅度隨轉角多寡而異：往返（180° 折返）
約 −13%、鋸齒約 −10%、矩形（90° 角、長邊）約 −2%、圓／8 字 <1%
（實際數字依 `lat_accel`／`accept_radius`／`jerk` 而變，`tests/smoke.py`
的 `naive` 欄可即時對照）。

---

## 障礙物與避障（箱子貼光球 → VRPN 讀進來 → 只繞不越）

場地裡的箱子貼上動捕光球、在 Motive 建成 **rigid body**，規劃器經 **VRPN** 讀取位置與姿態，
把箱子當障礙物畫在 3D 預覽裡，並在產生路徑時避開：**箱頂 ≤ 0.6 m 的低矮箱子局部拉高越過**、
較高的（疊放、放桌上）**水平繞開**（繞開時障礙物視為無限高的垂直柱；`low_mode: around` 可改成一律繞開）。
已知座標也可直接寫在設定檔 / GUI 手動輸入並隨設定儲存。

- **來源**
  - GUI 右欄「障礙物」頁：新增箱子（中心 x/y、底部 z、長寬高、偏航）或光球組（每行 `x y z`，
    XY 凸包當底面、最高光球當頂）；「從 VRPN 讀取」連上 Motive 的 VRPN 串流（`Data Streaming →
    VRPN`，預設 `localhost:3883`）收 2 s，每個 rigid body 變成一個箱子（偏航取自姿態；`pivot`
    指定樞紐點在箱頂 / 中心 / 底），或 `hull` 模式把所有 rigid body 位置合成一個障礙物
    （每個角各放一組光球時用）。VRPN 項目會取代上次讀進來的，手動項目保留。
  - **同一種箱子有躺有立**：`box_size` 填箱子三邊（例 35×40×50 cm），`orientation: auto_square`
    （預設）用量到的頂面高度判斷哪一邊垂直、底面取另外兩邊（VRPN 看不出哪一邊沿 x，所以保守取較長邊
    的正方形；`auto_long_x` / `auto_long_y` 指定長邊方向；`fixed` 照 x y z 順序）。個別箱子可用
    `sizes: {box3: [0.50, 0.40, 0.35]}` 指定；讀進來的項目也可以在清單裡直接改長寬高。
  - CLI：`python main.py --cli vrpn [--server host:port] [--save scene.yaml]` 列出 tracker
    （標明哪個是箱子、哪個是飛機），轉成障礙物與起飛點並存成 YAML，之後 `--config scene.yaml`
    疊加使用；`--cli list` 會列出設定檔裡的起飛點與障礙物。
  - **座標系**：障礙物座標必須與飛控 EKF 原點（= 房間 mocap 原點）同一座標系。Motive 預設
    Y-up，`obstacles.vrpn.axes: y_up` 轉成 ENU (x, −z, y)；Motive 改 Z-up 串流則設 `z_up`；
    也可自訂 `'x,-z,y'`，`offset` 可再平移。請與你的 mocap→飛控橋接用同一種轉換。
  - VRPN 客戶端是純 Python（`core/vrpn_client.py`，不需 vrpn 原生綁定），只讀 TCP、
    不必先知道 tracker 名稱；已對真實 Motive（VRPN 協定 07.33，Broadcast Port 3883）驗證可收到
    rigid body。**VRPN 只串流 rigid body，不串流個別光球**：每個箱子上的光球要在 Motive 建成一個
    rigid body（名稱就是 tracker 名稱）。無人機自己的 rigid body 用 `exclude`（GUI「排除 tracker」）
    略過，才不會被當成障礙物。連不上或看不到 rigid body 時先用 `--cli vrpn` 檢查。
- **機身尺寸**：`obstacles.drone_size`（預設 `[0.50, 0.50, 0.15]` m，長 × 寬 × 高）。規劃出來的是
  **機身參考點**的路徑，機身在它周圍佔一塊體積：
  - **水平**：機頭方向隨軌跡改變，所以會先碰到東西的是離中心最遠的角，也就是**半對角**而不是半寬。
    50×50 cm 的半寬 0.25 m、半對角 0.354 m，差的這 10 cm 就是撞到與否。安全檢查會擋下 `clearance`
    或 `margin.wall` 小於半對角的設定，並在低於建議值（半對角 + 航點切角 + 0.10 m 追蹤餘裕，
    50 cm 機身為 0.50 m）時警告。`clearance: auto` 直接用建議值。
  - **垂直**：規劃與飛控回報的都是**參考點**的高度，而參考點就是光球群的中心。光球全貼在頂板時
    參考點在機身上緣，**整個機體掛在規劃高度下方**，往下的餘裕要從 `obstacles.marker_height`
    （停在地上時光球離地高度）扣，不是從半高扣；往上只剩光球本身（`marker_above`，預設 3 cm）。
    這兩個值分別用於 `margin.floor` / `margin.ceiling`、越過箱子的 `vertical_clearance`
    （建議 ≥ 下方機體 + 0.15 m 下洗餘裕），以及起飛爬升柱的頂端（起飛高度 + 上方機體 + 0.05 m 超調）。
    `marker_height` 留空則退回「參考點在機身中心」的假設。
- **避障演算法**（`core/obstacles.py`）：底面外擴 `clearance`（預設 0.5 m，見上面的機身尺寸；圓角以每 90° `corner_segments` 段折線外接近似，保證離障礙物外緣 ≥ clearance）成禁區 →
  重疊 / 相碰的禁區合併成凸包（縫 < 2×clearance 本來就不能穿）→ 裁到安全盒 → 對 pattern 單圈折線：
  落在禁區內、或落在禁區與牆封住的口袋裡（從 HOME 到不了）的頂點推到禁區邊界上前後最省路的點
  （曲線 pattern 直接拿掉這些密集點）；穿過禁區的線段以**可視圖最短路**沿禁區的邊繞過去，只多出
  必要的轉角（航點越少越好）。改道後交回原本的圈數 / 速度剖面 / 高度 = f(弧長) / 航點匯出流程，
  所以 DO_JUMP 每圈相同、AUTO 工時估算與匯出折線一致。
- **低矮箱子越過、高的繞開**（`obstacles.low_mode: over`，預設）：箱頂 ≤ `over_max_top`（預設 0.6 m，
  單一個箱子放地上）且箱頂 + `vertical_clearance`（預設 0.5 m）不超過天花板邊界的箱子不做水平改道；
  路徑進入其禁區（底面外擴 clearance）時高度強制拉到箱頂 + 垂直安全距離（平台），前後以
  `over_ramp_speed`（auto = 與 stair 模式相同的安全垂直速度）的斜坡銜接，其餘位置維持原高度剖面
  （高度仍是弧長的函數，DO_JUMP 每圈相同）。兩個平台間隙太短就合併成一個平台。疊起來、放桌上的
  箱子（頂面超過門檻）仍水平繞開；`low_mode: around` 則一律繞開。匯出航點時平台起訖與斜坡轉折點會
  強制成為航點（直線與 SPLINE 都是），否則稀疏航點之間的直線內插會切到平台下方；安全檢查改為 3D：
  在越過箱子的禁區內高度必須 ≥ 箱頂 + 垂直安全距離（離箱頂 < 0.15 m 直接 ERROR）。
  > 注意下洗氣流：空紙箱很輕，從上方 0.5 m 通過可能把箱子吹動，箱子一動 rigid body 位置就變了；
  > 越過前請把箱子壓重，或改用 `around`。
- **起飛點（飛機實際停放的位置）**：`waypoints.takeoff_point` 是 `origin`（房間原點，預設）或
  `[x, y]`。AUTO 上傳後，飛機是從**它實際停放的地方**起飛再飛向第一個航點的，所以這一段必須用
  真實位置規劃，否則航點檔看起來正常，第一段卻可能直接穿過箱子。取得方式：
  - GUI「障礙物」頁的「從 VRPN 讀取」一次讀完所有 rigid body，箱子變成障礙物，名為
    `obstacles.vrpn.drone`（預設 `drone_01`）的那一個變成起飛點（可用「讀取時一併更新起飛點」關掉）。
    飛機永遠不會被當成障礙物。
  - CLI：`python main.py --cli vrpn --save scene.yaml` 會把 `obstacles.items` 與
    `waypoints.takeoff_point` 一起存出來，之後 `--config scene.yaml` 疊加。
  - 讀取是**按鈕觸發**，不是每次規劃都連線，所以規劃結果可重現、離線也能跑。
  起飛點會用在：進場航點的繞行起點、AUTO 工時估算的第一段、起飛垂直爬升檢查、RTL 檢查、障礙物可達性
  錨點（障礙物把房間切成兩半時，路徑會留在飛機那一半），以及匯出檔第 0 列 HOME 的經緯度。
- **起飛垂直爬升檢查**：飛機在起飛點原地爬到 `takeoff_alt`。只看水平距離不夠，因為箱子可能疊高或
  放在桌上（例如底 0.92 m、頂 1.42 m），飛機爬到 1.0 m 正好卡進去。所以同時比對水平距離與 z 區間：
  水平距離小於安全距離且箱底不高過起飛高度就是 ERROR；整個高過爬升柱的則只警告。
- **進場段**：AUTO 從起飛點起飛後飛到第一個航點若被擋住（可越過的低箱也算，進場高度只有起飛
  高度），匯出時會在 DO_JUMP block 之前插入
  進場航點（只飛一次，估時與精簡預算都算進去）；GUIDED 串流的「接近起點」也沿同樣的繞行路線。
  RTL 不會避障，結束動作有障礙物時建議 `land`。
- **安全檢查**：軌跡與實際匯出的 AUTO 航點折線（含進場段）進入障礙物底面 → ERROR；離障礙物 <
  clearance → WARN（曲線 pattern 的 SPLINE 航點之間是弦，會略切禁區圓角；實飛樣條介於弦與規劃
  路徑之間，要保證距離就加大 clearance 或縮小航點間距）；繞不開、HOME 在障礙物內也會 ERROR。
  `obstacles.enabled: false` 只顯示不改道（穿過仍會被擋下）。
- 預覽：箱子畫成半透明紅色柱體、光球為黃點，地板上的橙色虛線是外擴禁區；統計欄顯示改道段數與
  路徑長度變化。設定在 `config/default.yaml` 的 `obstacles` 區。

---

## 安裝

```powershell
cd C:\Users\admin\Documents\indoor_traj_studio
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

需求：Python ≥ 3.10、PyQt6、numpy、matplotlib、pymavlink、PyYAML。

---

## 快速開始

### GUI

```powershell
python main.py
```

左欄選軌跡 → 右欄調參數 → 按「產生 / 預覽」看 3D 軌跡與統計（工時 / 長度 / 速度 /
高度範圍 / 航點數 / 安全檢查）→ 用工具列匯出 `.waypoints` / `CSV` / `PNG`，
或「連線並飛行 (GUIDED)」。

### CLI（headless）

```powershell
python main.py --cli list                         # 列出軌跡與可用盒
python main.py --cli preview circle               # 存 3D 預覽 PNG
python main.py --cli export all                   # 匯出全部軌跡 (.waypoints + csv)
python main.py --cli export circle --formats waypoints,csv,plan,png
python main.py --cli fly circle                   # GUIDED 乾跑 (不解鎖)
python main.py --cli fly circle --confirm         # GUIDED 真的起飛 (謹慎!)
python main.py --cli vrpn --save scene.yaml        # 從 Motive VRPN 讀箱子 + 飛機位置 (障礙物 + 起飛點)
python main.py --cli --config scene.yaml export all # 帶著場景匯出 (--config 要放在子命令前)
python main.py --obstacles                        # GUI 直接開在「障礙物」頁
```

輸出預設在 `output/`。自訂設定：`--config my.yaml`（深層覆寫 `config/default.yaml`）。

---

## 輸出格式

- **`.waypoints` (QGC WPL 110) — 雙通道**。第 0 列 HOME 寫的是 `waypoints.takeoff_point` 換算的
  經緯度（未設定時等於假原點，與舊版逐字元相同）；飛控真正的 home 仍由 mocap 橋接的
  `SET_HOME_POSITION` 或解鎖位置決定，這一列主要讓 Mission Planner 的地圖對得上實際停機位置。
  序列：`HOME → TAKEOFF → DO_CHANGE_SPEED →
  NAV_WAYPOINT / NAV_SPLINE_WAYPOINT × N → LAND/RTL`。欄位排列與 aeroplan-studio
  `mission/waypoint.py` 一致；室內精度需求高，故 lat/lon 用 **8 位小數 (~1.1 mm)**。
  **航點越少、每段越長，實飛速度才跑得滿**：ArduPilot 4.1+ 每段航線各自算「靜止→靜止」的
  S-curve 再依前後段修剪，一段能達到的峰值速度受段長限制——0.25 m 的段在保守參數
  (accel 1、jerk 1) 下峰值只有 0.25 m/s、原廠參數 (2.5 / 5) 也只有 0.43 m/s，都到不了 0.5 巡航；
  約 ≥ 1 m 的段才跑得滿。因此匯出預設盡量少放航點：
  - **`waypoints.repeat: do_jump`（預設）** — 只寫**一圈**的航點，後面接
    `DO_JUMP(回第一個航點, 重複 圈數−1 次)` 與收尾點（回起點）再 LAND，任務項數與圈數無關
    （往返 sine 14 圈 = 22 項；振幅 0 = 8 項）。代價是每一圈高度必須相同：起伏次數會自動調成
    **每圈整數次**（`max(1, round(cycles/圈數))`，GUI 統計欄會顯示實際值）；`unroll` = 全部
    圈數展開（各圈高度相位可錯開，但航點數 × 圈數）。
  - **高度是「沿路徑弧長」的函數**（不是時間）——與 AUTO 航點內插行為一致，也是 DO_JUMP 每圈
    完全相同的前提；GUIDED 軌跡以 z(s(t)) 取樣，兩者一致。
  - **直線型軌跡（往返 / 矩形 / 鋸齒）預設「精確轉角 + 高度容差精簡」**（`waypoints.sparse_straight`
    + `z_tol`，預設 2 cm）：以精確轉角為航點，段內只要折線與高度剖面的差 ≤ `z_tol` 就拿掉中間點
    ——振幅 0 時往返 = `A,B,A,B…` 兩點一直線、矩形 = 每圈 4 角；線性升降（triangle、stair 過渡）
    只留折點；sine 等曲線的段長由 `z_tol` 決定（越大航點越少、段越長）。`z_tol: 0` = 不精簡。
  - 曲線（圓 / 8 字）沿弧長依 `point_spacing × spline_spacing_mult` 取 SPLINE 航點（每圈等分，
    不會有零頭短段）。
  - **到航點停下 → 原地轉頭 → 再前進（`waypoints.turn_in_place`，預設關閉）** — 見下一節。
  匯出時一次產生兩個檔：
  - **`<軌跡>.waypoints` (高精度)** — 依上述規則取樣（`repeat: unroll` 且不精簡時點數可能很多），
    給分析 / 模擬 / 對照用。AUTO 工時預估與匯出用同一條（實飛）折線。
  - **`<軌跡>_compact.waypoints` (精簡 / 上飛控)** — 保證 **總任務項 ≤ `fc_budget`(預設 650)**，
    確保寫得進飛控；用等弧長重取樣壓到預算內。若高精度版本來就 ≤ 預算，兩檔相同。
- **`.csv`** — 全程時間序列 ground truth：`t, x_e,y_n,z_u, vx,vy,vz, yaw_deg,
  north,east,down, lat,lon,rel_alt`（ENU + NED + WGS84），20 Hz。給 VIO / 預測模型用。

### 到航點原地轉頭（`MAV_CMD_CONDITION_YAW`）

開關：`waypoints.turn_in_place.enabled`（GUI「匯出 .waypoints」群組的勾選框；CLI 加 `--turn-in-place`，
要放在子命令前）。開啟後，**水平轉向角 ≥ `min_turn_deg`（預設 30°）的航點**在任務裡變成：

```
NAV_WAYPOINT(A, hold 0) → CONDITION_YAW(115: 航向, 角速度, 方向, 0=絕對) → NAV_DELAY(93: 秒, -1, -1, -1) → NAV_WAYPOINT(B)
```

飛機在 A **完全停下 → 原地轉頭朝 A→B 的方向 → NAV_DELAY 秒後才飛向 B**。DO_JUMP block 裡的轉頭點每圈都會轉；
起飛到高度後也會先原地轉向第一段（`after_takeoff: true`）。只影響 AUTO 任務，GUIDED 串流與 CSV 不變。

- **為什麼這樣排**（對照 ArduPilot `AP_Mission.cpp` / `ArduCopter/mode_auto.cpp`，Copter-4.4、4.5、master 皆同）：
  - DO/CONDITION 命令在**前一個導航命令完成（到點）時**才開始，與下一個導航命令並行 → CONDITION_YAW 放在 A 後面
    就是「到 A 才轉」。
  - 下一個導航命令是 NAV_DELAY 時，A **不做 fast waypoint**（S-curve 在 A 減速到 0），NAV_DELAY 期間不改 submode、
    不改 yaw → 飛機懸停在 A 轉頭。
  - NAV_DELAY 結束、B 開始時，未轉完的 CONDITION_YAW 會被丟掉、yaw 交回 `WP_YAW_BEHAVIOR` → **NAV_DELAY 必須夠長**。
    延遲 = 要轉的角度 / min(`rate_deg_s`, 60°/s) + `settle_s`（進位到 0.1 s）。60°/s 是 `ATC_SLEW_YAW`（新版
    `ATC_RATE_WPY_MAX`）的預設上限，CONDITION_YAW 的角速度不會超過它。
  - 不用「航點 hold time（p1）」：CONDITION_YAW 會等 hold 結束、開始飛向 B 時才開始 → 變成邊飛邊轉；hold 也只能整數秒。
  - **NAV_DELAY 的 p1 絕不能 ≤ 0**：ArduCopter 會改成「等到 UTC 時:分:秒」（p2..p4 = 0 時是等到 00:00 UTC）。
    本程式一律寫 p1 ≥ 0.5 s、p2..p4 = −1。
- **航向**：CONDITION_YAW p1 是絕對航向（0 = 北、順時針），這裡的「北」= 房間 +y（與航點經緯度換算同一個假原點
  座標系），因此 mocap 橋接給 EKF 的 yaw 必須與房間座標對齊。方向一般用最短（p3 = 0）；原路折返（≈180°，最短方向
  不明確）改用 `u_turn_dir`（cw / ccw），但若會讓 DO_JUMP 某一圈繞遠路就退回可行的方向。
- **飛控參數**：`WP_YAW_BEHAVIOR` 請維持 **2**（預設，朝下一航點）——轉完後飛向 B 時機頭已經對準，不會再轉。
  Copter-4.4/4.5 不要用 0（未轉完的轉頭會整段凍結），室內也不要用 3（要地速 > 1 m/s 才更新）。
  Copter-4.4 在每段起步的瞬間可能有一下小 yaw 抖動（速度 < 5% WPNAV_SPEED 時沿用舊航向）。
- **估時與圈數**：AUTO 估時會在轉頭點把航線切開（靜止 → 靜止）再加上懸停秒數，自動圈數會跟著減少；轉頭點多的
  pattern（鋸齒、往返）航線時間明顯變長，一圈就超過工時上限時會照常警告（可調高門檻、角速度或降低 `settle_s`）。
  實機到點還要先收斂到 `WPNAV_RADIUS` 內才開始轉，若實飛比預估長，加大 `settle_s`。
- **任務項數**：每個轉頭點 +2 項（DO_JUMP 重複不另計），精簡版預算會把它們算進去。
- **曲線 pattern**（圓 / 8 字 / 隨機手飛）預設不在曲線航點轉（`curves: false`；SPLINE 航點停下會破壞曲線），
  進場繞障航點與起飛後轉向照常。
- 3D 預覽以橘色三角形標出轉頭點；GUI 統計欄 / CLI 摘要顯示轉頭點數、實飛停下次數與懸停總秒數。

---

## 兩種在 ArduPilot 上飛的方式

### A) AUTO（上傳 `.waypoints`）
用 Mission Planner 載入 **`output/<pattern>_compact.waypoints`**（已壓到 ≤650 項，確保寫得進
飛控）→ 上傳 → 切 AUTO。需要最高保真的對照資料才用 `<pattern>.waypoints`（高精度版）。
直線軌跡用一般航點，**曲線 (圓/8字) 用 SPLINE 航點** 較平滑。
> 注意：AUTO 對小空間曲線會有「切角」與接受半徑限制；要最平滑請用 B)。

### B) GUIDED 即時串流（推薦給曲線）
`runner/guided_runner.py` 以 ~20 Hz 串流 `SET_POSITION_TARGET_LOCAL_NED`
（`MAV_FRAME_LOCAL_NED`，ENU→NED：N=y, E=x, D=−z），流程：
`GUIDED → arm → takeoff → 平滑接近起點 → 串流軌跡 → LAND`。曲線最平滑、不受切角影響。
Ctrl+C 會切 LAND。

---

## 室內非 GPS / 定位整合（重點）

**本程式不含 mocap→飛控 橋接**（你已有，例如 `vision_to_mavros` 或自製 NatNet 橋接）。
以下為飛控端「外部定位」對照參數（ArduCopter 4.x，OptiTrack/Vicon 非 GPS 導航；
請依你既有橋接微調）：

```
AHRS_EKF_TYPE = 3        # 用 EKF3
EK3_ENABLE    = 1
EK2_ENABLE    = 0
GPS_TYPE      = 0        # 無 GPS (走 VISION_POSITION_ESTIMATE)
EK3_SRC1_POSXY = 6       # ExternalNav
EK3_SRC1_VELXY = 6       # ExternalNav (有送速度才設)
EK3_SRC1_POSZ  = 6       # ExternalNav (或 1=Baro)
EK3_SRC1_YAW   = 6       # ExternalNav (mocap 提供 yaw)
VISO_TYPE      = 依你的橋接 (raw VISION_POSITION_ESTIMATE 通常可不設)
COMPASS_USE    = 0       # 若完全用 mocap yaw
```

橋接需先送 `SET_GPS_GLOBAL_ORIGIN`（與 `SET_HOME_POSITION`）設定 EKF 原點，EKF 才會
輸出本地位置；位置訊息建議 ≥ 10 Hz（EKF 至少需 4 Hz）。

**座標對齊（很重要）**
- **GUIDED (B)**：設點用 `LOCAL_NED`，相對 EKF 原點。只要 **房間 (mocap) 原點 = EKF 原點**，
  軌跡就會落在正確的房間座標，免設經緯度。
- **AUTO (A)**：航點是繞 `config` 的假原點 `origin_lat/origin_lon` 換算的經緯度，
  因此請把 `origin_lat/origin_lon` 設成 **與橋接 `SET_GPS_GLOBAL_ORIGIN` 相同的經緯度**，
  AUTO 軌跡才會對應到正確房間位置。

---

## 用 SITL 測試（不接真機）

1. 啟動 ArduCopter SITL（Mission Planner 內建或 `sim_vehicle.py -v ArduCopter`），
   預設 `tcp:127.0.0.1:5760`。
2. 設好上述外部定位參數（SITL 也可灌 mocap/假定位來源）。
3. 乾跑檢查連線：`python main.py --cli fly circle`
4. 實際飛：`python main.py --cli fly circle --confirm`
   或在 GUI 按「連線並飛行 → 真的起飛」。
5. AUTO 比對：把 `output/circle.waypoints` 上傳 SITL 切 AUTO。

---

## 安全須知

- `--confirm` / 「真的起飛」才會解鎖飛行；其餘為乾跑。飛行前確認 **場地淨空、設安全圍籬**。
- 內建安全檢查：軌跡超出安全盒、超速、工時不足、航點數過多都會擋下或警告。
- 預設安全邊界：離牆 0.5 m、離地/天花板 0.6 m（5×5×3 → 可用 4×4×1.8 m，z∈0.6~2.4）。

---

## 設定檔（`config/default.yaml`）

`volume`（空間）、`margin`（安全邊界）、`flight`（速度/工時/圈數）、`altitude`（高度調變）、
`yaw`（朝向）、`waypoints`（匯出/假原點/間距/SPLINE）、`guided`（連線/串流）、
`obstacles`（障礙物 / 機身尺寸 / 安全距離 / VRPN 設定）、`patterns`（各軌跡形狀微調）。
GUI 可即時改並「儲存設定」。起飛點在 `waypoints.takeoff_point`，飛機的 rigid body 名稱在
`obstacles.vrpn.drone`，機身尺寸在 `obstacles.drone_size` 與 `obstacles.marker_height`。

## 專案結構

```
core/        幾何、軌跡、圖形(patterns)、高度、工時、安全、planner
             obstacles.py 障礙物 / 禁區 / 可視圖避障, vrpn_client.py 純 Python VRPN 客戶端
io_export/   .waypoints / csv 匯出
runner/      GUIDED 即時串流 (pymavlink)
ui/          PyQt6 GUI + 3D 預覽 (obstacle_panel.py 障礙物頁)
viz.py       共用 matplotlib 3D 繪圖
cli.py       命令列
tests/       smoke.py 核心煙霧測試, obstacles_smoke.py 障礙物 / 避障 / VRPN 測試
```
