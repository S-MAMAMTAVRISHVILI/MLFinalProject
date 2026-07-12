# Walmart Recruiting — Store Sales Forecasting

მაღაზიების დეპარტამენტების ყოველკვირეული გაყიდვების პროგნოზირება (Kaggle-ის competition:
[Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)).
ეს არის **Time-Series** ამოცანა, სადაც ვამუშავებთ სხვადასხვა არქიტექტურის მოდელს — Tree-Based,
Deep Learning, Classical Statistical და Foundation — და ვადარებთ, რომელმა მიაღწია უკეთეს შედეგს
და **რატომ**.

პროექტი შესრულებულია 2-კაციან გუნდში. ეს README აღწერს პროექტის ყველა ეტაპს: მონაცემთა ანალიზს,
preprocessing-ს, ვალიდაციის სტრატეგიას, feature engineering-ს, თითოეული მოდელის მიდგომასა და
შედეგს, საბოლოო არჩევანსა და დასკვნებს.

---

## სარჩევი

1. [ამოცანა და შეფასების მეტრიკა](#1-ამოცანა-და-შეფასების-მეტრიკა)
2. [რეპოზიტორიის სტრუქტურა](#2-რეპოზიტორიის-სტრუქტურა)
3. [მონაცემები და EDA](#3-მონაცემები-და-eda)
4. [Preprocessing — მონაცემთა დამუშავება](#4-preprocessing--მონაცემთა-დამუშავება)
5. [ვალიდაციის სტრატეგია](#5-ვალიდაციის-სტრატეგია)
6. [Feature Engineering](#6-feature-engineering)
7. [მოდელები და შედეგები](#7-მოდელები-და-შედეგები)
8. [შედეგების საბოლოო შედარება](#8-შედეგების-საბოლოო-შედარება)
9. [საუკეთესო მოდელი და Model Registry](#9-საუკეთესო-მოდელი-და-model-registry)
10. [MLflow-ის სტრუქტურა](#10-mlflow-ის-სტრუქტურა)
11. [ძირითადი დასკვნები](#11-ძირითადი-დასკვნები)
12. [გუნდის განაწილება](#12-გუნდის-განაწილება)
13. [გაშვების ინსტრუქცია](#13-გაშვების-ინსტრუქცია)

---

## 1. ამოცანა და შეფასების მეტრიკა

45 მაღაზიისთვის, თითოეულში მრავალი დეპარტამენტით, უნდა ვიწინასწარმეტყველოთ **ყოველკვირეული
გაყიდვები** მომდევნო 39 კვირისთვის. თითოეული პროგნოზი შეესაბამება `(Store, Dept, Date)` სამეულს.

შეფასების მეტრიკა არის **WMAE** (Weighted Mean Absolute Error):

```
WMAE = ( Σ wᵢ · |yᵢ − ŷᵢ| ) / Σ wᵢ ,      wᵢ = 5, თუ კვირა სადღესასწაულოა; სხვა შემთხვევაში 1
```

ორი კრიტიკული დასკვნა ამ მეტრიკიდან, რომლებმაც განსაზღვრეს ჩვენი მთელი მიდგომა:

* **ეს არის L1 (აბსოლუტური) მეტრიკა.** ამიტომ ყველა მოდელს ვავარჯიშებთ **MAE / absolute-error**
  objective-ით და არა RMSE-ით. (გადამოწმებული: per-pair median → WMAE 2419, per-pair mean → 2628
  `recent` fold-ზე; median, რომელიც MAE-ოპტიმალურია, უკეთესია.)
* **სადღესასწაულო კვირები დომინირებენ.** test-ში ისინი მწკრივების მხოლოდ 7.76%-ია, მაგრამ WMAE-ის
  წონის **29.6%-ს** ატარებენ. ამიტომ ვიყენებთ `sample_weight = 5` სადღესასწაულო კვირებზე.

---

## 2. რეპოზიტორიის სტრუქტურა

```
MLFinalProject/
├── src/
│   ├── walmart_prep.py      # საერთო preprocessing: WalmartFeatureBuilder, WMAE, folds, residual target, December gate
│   ├── walmart_panel.py     # dense panel sequence-მოდელებისთვის (DLinear, N-BEATS, PatchTST, TFT)
│   └── walmart_dl.py        # DLinear / N-BEATS / PatchTST არქიტექტურები + SeqForecaster + სავარჯიშო loop
├── model_experiment_LightGBM.ipynb
├── model_experiment_XGBoost.ipynb
├── model_experiment_DLinear.ipynb
├── model_experiment_ARIMA_SARIMA.ipynb
├── model_experiment_TimesFM.ipynb
├── model_experiment_NBEATS.ipynb
├── model_experiment_PatchTST.ipynb
├── model_experiment_Prophet.ipynb
├── model_inferenceV1.ipynb    # საუკეთესო მოდელს იღებს Model Registry-დან და აგენერირებს submission-ს
└── submissions/                # Kaggle submission ფაილები
```

**საერთო კოდის პრინციპი:** ყველა notebook აიმპორტებს `src`-ს, ამიტომ ყველა მოდელი
ერთსა და იმავე feature-ებზე, ვალიდაციის fold-ებზე და საერთო მეტრიკით
ფასდება.

---

## 3. მონაცემები და EDA

| ფაილი | მწკრივები | პერიოდი |
|---|---|---|
| `train.csv` | 421,570 | 2010-02-05 → 2012-10-26 (143 კვირა) |
| `test.csv` | 115,064 | 2012-11-02 → 2013-07-26 (39 კვირა) |
| `features.csv` | 8,190 | 45 მაღაზია × 182 კვირა (Temperature, Fuel_Price, MarkDown1–5, CPI, Unemployment) |
| `stores.csv` | 45 | მაღაზიის ტიპი (A/B/C) და ზომა |

**მთავარი სტრუქტურული ფაქტები:**

* კვირები მთავრდება **პარასკევს**, ზუსტად 7-დღიანი ინტერვალით.
* 45 მაღაზია × 81 დეპარტამენტი → train-ში 3,331 შესწავლილი `(Store, Dept)` წყვილი, test-ში 3,169.
* **test არ არის შემთხვევითი split.** ეს არის ტრენინგის დასრულების მომდევნო 39 კვირა. ეს ერთი
  ფაქტი განსაზღვრავს დანარჩენ თითქმის ყველა გადაწყვეტილებას (იხ. [ვალიდაცია](#5-ვალიდაციის-სტრატეგია)).

**სამიზნე ცვლადი (`Weekly_Sales`):** მარჯვნივ დახრილი (skew 3.26, kurtosis 21.5), median $7,612,
mean $15,981, max $693,099. მწკრივების **0.305% უარყოფითია** (min −$4,988) — ეს რეალური returns/
შესწორებებია და შენარჩუნებულია. ამის გამო `log`/`log1p` პირდაპირ ვერ გამოიყენება.

**სეზონურობა (ავტოკორელაცია):** გაყიდვები ბევრად უფრო ძლიერად კორელირებს **1 წლის წინანდელ**
მნიშვნელობასთან, ვიდრე წინა კვირასთან:

| lag | 1 | 4 | 26 | 51 | **52** | 53 |
|---|---|---|---|---|---|---|
| corr | 0.948 | 0.933 | 0.887 | 0.946 | **0.983** | 0.949 |

`lag_52` მთელი მონაცემთა ბაზის ყველაზე ინფორმაციული სვეტია — ეს არის მთელი feature engineering-ის
საფუძველი.

**ხარვეზები / cold-start:** 605 სერიას აქვს შიდა ხარვეზი; 11 წყვილი test-ში არასდროსაა train-ში
(36 მწკრივი); 25 წყვილს არ აქვს დაკვირვება ბოლო 52 კვირაში (78 მწკრივი). ეს რიცხობრივად უმნიშვნელოა,
მაგრამ pipeline არ უნდა ჩავარდეს მათზე.

---

## 4. Preprocessing — მონაცემთა დამუშავება

ყველა წესი იმპლემენტირებულია `WalmartFeatureBuilder`-ში (sklearn transformer), რომ **fit მოხდეს
train-ზე და transform პირდაპირ დაუმუშავებელ `test.csv`-ზე** — ეს არის Pipeline-ის მოთხოვნის
შესრულების საფუძველი. თითოეული გადაწყვეტილება ემპირიულადაა დასაბუთებული:

| პრობლემა | გადაწყვეტა | დასაბუთება |
|---|---|---|
| **შიდა ხარვეზები** | შევსება 0-ით | ხარვეზის მეზობელი გაყიდვების median $11.4, და 94.5% < $500 — ე.ი. დეპარტამენტი ფაქტობრივად არ ყიდდა |
| **MarkDown1–5 (50–64% NA)** | შევსება 0 + `markdown_era` flag | ორი განსხვავებული მიზეზი: 2011-11-11-მდე საერთოდ არ იწერებოდა; მას შემდეგ NA ნიშნავს „markdown არ ყოფილა". მხოლოდ train-ის **35.9%** არის markdown-ის ეპოქაში, test-ის **100%** |
| **CPI / Unemployment (585 NA)** | forward-fill + გადახრა მაღაზიის საშუალოდან | NA არის test-ის ბოლო 13 კვირა (reporting lag). test-ის CPI-ის **54%** სცდება train-ის მაქსიმუმს → ხეები ვერ extrapolate-ენ; ამიტომ raw-ის ნაცვლად გადახრას ვიყენებთ |
| **უარყოფითი გაყიდვები** | შენარჩუნება | მხოლოდ 0.305%, ნამდვილი returns |
| **Store / Dept** | numeric (არა categorical) | categorical split-ის ძებნა 81 დეპარტამენტზე overfit-ს იწვევდა (ყველა fold-ზე უარესი) |
| **target transform** | არა (log1p არა) | WMAE არის L1; log1p ცვლის loss-ის ფორმას. გამოსაცდელი, არა ჩავარდნილი დაშვება |

---

## 5. ვალიდაციის სტრატეგია

რადგან test არის 39-კვირიანი ბლოკი ტრენინგის შემდეგ, ვალიდაციაც უნდა იყოს **39-კვირიანი
rolling-origin** ბლოკები. ავირჩიეთ 3 fold:

| fold | train სრულდება | ვალიდაცია | სადღესასწაულო კვირები | naive WMAE |
|---|---|---|---|---|
| **`mirror`** | 2011-10-28 | 2011-11-04 → 2012-07-27 | Thanksgiving, Christmas, Super Bowl | **2037.8** |
| `recent` | 2012-01-27 | 2012-02-03 → 2012-10-26 | Super Bowl, Labor Day | **1807.2** |
| `early` | 2011-07-29 | 2011-08-05 → 2012-04-27 | Labor Day, Thx, Xmas, SB | **2018.6** |

**`mirror` არის მთავარი (selection) fold.** მისი ვალიდაციის ფანჯარა კალენდარულად ზუსტად
ემთხვევა რეალურ test-ს ერთი წლით ადრე (11-04→07-27 vs 11-02→07-26), ამიტომ მას აქვს იგივე
სადღესასწაულო შემადგენლობა და სადღესასწაულო წონის წილი **29.7%** (test-ის 29.6%-ის წინააღმდეგ).
`recent` და `early` — მეორე აზრი; ვაქვეყნებთ სამივეს.

**კრიტიკული წესი — მინიმალური „უსაფრთხო" lag = 39.** test-ის ბოლო კვირისთვის `lag < 39` უცნობია
(ან recursion-ს საჭიროებს 39 ნაბიჯით დაგროვებული შეცდომით, ან leakage-ს იწვევს). ამიტომ
`WalmartFeatureBuilder` **უარყოფს ნებისმიერ lag < 39-ს**. გამოსაყენებელი დიაპაზონი: 39 … 104.

**seasonal naive** (იწინასწარმეტყველე იგივე `(Store, Dept)` 52 კვირით ადრე) — ეს არის რიცხვი,
რომელიც ყველა მოდელმა უნდა დაამარცხოს. ეს არ არის სუსტი baseline: `lag_52` corr = 0.983.

---

## 6. Feature Engineering

`WalmartFeatureBuilder.transform()` აბრუნებს 52 სვეტს. მთავარი ჯგუფები:

* **იდენტობა/სტატიკური:** `Store`, `Dept`, `Type`, `Size`, `IsHoliday`
* **კალენდარული** (ცნობილი ყველა მომავალი კვირისთვის): `woy_sin/cos`, ნიშნიანი მანძილები
  `days_to_{xmas,thanksgiving,superbowl,laborday,easter}`, `pre_xmas_days`, `is_*_week`
* **გაყიდვის lag-ები (≥ 39):** `lag_{39,45,52,53,104}`
* **წლის-წინანდელი ფანჯრის აგრეგატები:** `roll_{mean,std,median}_lag52_w5`, `level_mean_lag39_90`, `yoy_trend`
* **ეგზოგენური** (ცნობილი test-ის მთელ ჰორიზონტზე): `Temperature`, `Fuel_Price`, `CPI_dev`, `Unemployment_dev`, `MarkDown1–5`

### 6.1. Residual target — ერთ-ერთი ყველაზე მნიშვნელოვანი გადაწყვეტილება

მოდელს ვავარჯიშებთ **ნაშთზე** `y − seasonal_baseline`, და არა თავად გაყიდვის დონეზე. მიზეზი:
გლობალური მოდელი, რომელიც raw დონეს პროგნოზირებს, ხარჯავს ტევადობას იმის სწავლაზე, რომ „მაღაზია 20,
დეპარტამენტი 92 დიდია" — ინფორმაცია, რომელსაც `lag_52` უკვე ზუსტად შეიცავს; ასევე ვერ აღწევს
სადღესასწაულო პიკებს. გაზომილი (HistGradientBoosting, `mirror` fold):

| მიდგომა | WMAE |
|---|---|
| seasonal naive | 2037.8 |
| **level** target | 2210.8 (naive-ზეც უარესი!) |
| **residual** target | **1822.5** |

Residual-მა Thanksgiving-კვირის MAE 5057-დან 2196-მდე ჩამოიყვანა. ეს იმპლემენტირებულია
`SeasonalResidualRegressor`-ში (ნებისმიერ sklearn მოდელს ახვევს Pipeline-ის შიგნით).

### 6.2. Christmas alignment პრობლემა — ამოცანის მთავარი სირთულე

კვირები პარასკევს მთავრდება, შობა კი — არა. ამიტომ „შობის კვირა" (`IsHoliday=True`) ყოველ წელს
სხვადასხვა რაოდენობის შობამდელ სავაჭრო დღეს შეიცავს:

| დროშით მონიშნული კვირა | ფარავს | Dec 22–24 შიგნით | ჯამური გაყიდვა |
|---|---|---|---|
| 2010-12-31 | Dec 25–31 | **0** | $40.4M (ფსკერი) |
| 2011-12-30 | Dec 24–30 | **1** | $46.0M |
| **2012-12-28** (test) | Dec 22–28 | **3** | *საპროგნოზო* |

ამასთან, 2010-12-24 და 2011-12-23 (არა-მონიშნული) მთელი მონაცემების ორი ყველაზე დიდი კვირა იყო
($80.9M, $77.0M). ამიტომ უბრალო `lag_52` 2012 წლის წონა-5 შობის კვირას ფსკერიდან კითხულობს.
**გამოსავალი — `xmas_aligned_lag`:** წინა წლის სერიას ვკითხულობთ **შობიდან იმავე მანძილზე**,
ორ მეზობელ კვირას შორის ინტერპოლაციით. ამან Christmas-კვირის MAE 33%-ით შეამცირა replay-ტესტში.

### 6.3. December gate — ვალიდაციის მიღმა შემოწმება

test-ის ყოველ დეკემბრის კვირას აქვს კალენდარული გეომეტრია, რომელიც **train-ში არასდროს გვხვდება**.
ამიტომ ვერცერთი ვალიდაციის fold ვერ დაიჭერს დეკემბრის შეცდომას. ამის გადასაჭრელად ვაშენებთ
**ყოველდღიური პროფილის დეკონვოლუციას** ორი დეკემბრის სეზონიდან (weekly აგრეგატებიდან ვაღდგენთ
დღიურ shape-ს) და ვამოწმებთ, პროგნოზი შეესაბამება თუ არა ფიზიკურად მოსალოდნელ ფორმას. `december_gate`
აქცევს მოდელს PASS/REVIEW-ად. ეს პროექტის ორიგინალური კონტრიბუციაა და ის, რაც ხეებს კლასიკური
მოდელებისგან განასხვავებს. მნიშვნელოვანი ფაქტი, რომელიც პროფილმა გამოავლინა: **2012 წლის დეკემბრის
პიკი არის 2012-12-21, და არა მონიშნული შობის კვირა 2012-12-28.**

---

## 7. მოდელები და შედეგები

თითოეული მოდელი ცალკე notebook-ია, ცალკე MLflow ექსპერიმენტით. ყველა notebook-ს აქვს ერთი და იგივე
სტრუქტურა: `Cleaning → Baseline → CV → Feature_Selection → Tuning → Final` (კლასიკურ/foundation
მოდელებზე შესაბამისად ადაპტირებული).

### 7.1. Tree-Based — LightGBM

* **მიდგომა:** residual target, `objective="l1"`, `sample_weight=5` სადღესასწაულო კვირებზე. Feature
  ablation (რომელი ბლოკი რას აძლევს) და random search `mirror` fold-ზე (incumbent trial-0-ად, რომ
  ძებნა default-ზე უარესს ვერ დააბრუნოს).
* **მნიშვნელოვანი აღმოჩენა:** `drop_time_index` ablation-მა აჩვენა, რომ `t` (დროის ინდექსი),
  მიუხედავად იმისა, რომ test-ის ყველა მწკრივზე train-ის ყველა split-ს სცდება, **მაინც სასარგებლოა** —
  ის ჰყოფს ეპოქებს (markdown-მდე/შემდეგ), და მისი მოცილება ყველა fold-ზე აზიანებდა.
* **შედეგი (tuned, trial_04):** mirror **1864.2**, recent **1650.3**, early **1910.0**, mean **1808.2**.
  December gate: **PASS** (პიკი 2012-12-21-ზე, სწორად). Registry version 2.

### 7.2. Tree-Based — XGBoost - საუკეთესო მოდელი

* **მიდგომა:** იგივე ინფრასტრუქტურა, `objective="reg:absoluteerror"`, `device="cuda"` (XGBoost-ის
  pip wheel-ს აქვს CUDA მხარდაჭერა, LightGBM-ისგან განსხვავებით). `max_depth=8` ≈ LightGBM-ის
  `num_leaves` დიაპაზონი.
* **შედეგი (tuned, trial_06 — `lr=0.03, depth=10, n=1500, sub=0.9, col=0.8`):**
  mirror **1817.7**, recent **1616.7**, early **1851.2**, mean **1761.9**. December gate: **PASS**.
  Registry version 3, alias `xgboost`.
* **ეს არის ამ ეტაპზე საუკეთესო შედეგის მქონე მოდელი** — ყველაზე დაბალი WMAE ყველა fold-ზე და gate PASS.

### 7.3. Deep Learning — DLinear

* **არქიტექტურა:** DLinear (Zeng et al., 2023) — სერიის დაშლა trend + seasonal-ად, თითოეულზე ერთი
  წრფივი ფენა [lookback → horizon]. channel-independent, საერთო წონებით. იყენებს `WalmartPanel`-ს
  (dense 3331×143 panel).
* **DL-ის ისტორიის შეზღუდვა:** `lookback=52 + horizon=39 = 91` კვირა სჭირდება ერთ window-ს.
  ამიტომ `early` fold საერთოდ ვერ იძლევა window-ს, `mirror` — მხოლოდ 1-ს სერიაზე. მხოლოდ `recent`
  და სრული მონაცემი ავარჯიშებს სრულფასოვნად. ეს DL-მოდელების „მონაცემზე შიმშილის" კონკრეტული ილუსტრაციაა.
* **მთავარი აღმოჩენა (residual study):** residual რეჟიმში ნულოვანი ქსელი **ზუსტად** naive-ს
  აღადგენს (0.0 სხვაობა ყველა მწკრივზე). weight_decay-ის ზრდისას WMAE **მონოტონურად ეშვება naive-ისკენ**
  და იქ ჩერდება: wd=0.0001→2391, wd=0.1→1947, wd=1.0→1816, wd=10→1808. ე.ი. წლიდან-წელს ცვლილება
  გლობალური წრფივი მოდელისთვის ხმაურია.
* **შედეგი:** best (residual, L=52, wd=1.0) recent **1816.2** ≈ naive. December gate: **REVIEW −24%**
  (ვერ ასწორებს შობის alignment-ს, რადგან `xmas_aligned_lag` არ აქვს). Registry version 4, alias `dlinear`.

### 7.4. Classical — ARIMA / SARIMA

ამ მოდელებზე დავალების ინსტრუქცია მოითხოვს **თეორიული ცოდნის მიღებას**. ამიტომ
notebook არის დიაგნოსტიკაზე ორიენტირებული:

* **სტაციონარობა:** ADF ტესტი აჩვენებს, რომ დონე არასტაციონარულია, პირველი სხვაობის შემდეგ —
  სტაციონარული. ACF-ს აქვს მკვეთრი პიკი **lag 52-ზე** (წლიური ციკლი).
* **ARIMA დემონსტრაცია:** საუკეთესო არა-სეზონური რიგი AIC-ით (3,1,2). ნაშთების ACF **მაინც ინარჩუნებს
  პიკს lag 52-ზე** — ე.ი. არა-სეზონური ARIMA სტრუქტურულად ვერ იჭერს წლიურ სეზონურობას.
* **SARIMA-ის იდენტიფიცირებადობა (მთავარი თეორიული პუნქტი):** SARIMA(1,1,1)(1,1,1,52)-ის ერთი fit
  6.4 წამია → სრული პანელისთვის ~358 წუთი. უფრო მნიშვნელოვანი: 143 კვირა = **მხოლოდ 2.75 სეზონური
  ციკლი**, ამიტომ სეზონური პარამეტრები **პრაქტიკულად იდენტიფიცირებადი არ არის** — `ar.S.L52` და
  `ma.S.L52` შეფასდნენ ≈0-ით, ხოლო confidence interval-ის სიგანით **~54**. ეს ცხადად აჩვენებს, რატომ
  არ არის SARIMA პრაქტიკული ამ მონაცემებზე.
* **სრული პანელის პროგნოზი (feasible proxy):** სეზონური differencing (lag 52) + დაბალი რიგის ARIMA
  → recent **1791.7** (naive-ს ოდნავ ჯობნის, +0.9%). December gate: **REVIEW −23.9%**. Registry version 5, alias `arima`.

### 7.5. Foundation — TimesFM (ბონუსი)

* **მიდგომა:** Google-ის **TimesFM 2.5 (200M)**, გამოყენებული **zero-shot** — ტრენინგის გარეშე.
  თითოეულ სერიას ვაძლევთ ისტორიას context-ად და 39 კვირას ვითხოვთ. `PrecomputedForecaster`-ით
  ერთვება იმავე raw-`test.csv` კონტრაქტში.
* **შედეგი — მოულოდნელად ძლიერი:** recent **1679.1** (**naive-ს ჯობნის 7.1%-ით!**). ეს **ჯობნის**
  როგორც ARIMA-ს (1791.7), ისე DLinear-ს (1816.2) — და ეს ტრენინგის გარეშე, მოდელით, რომელსაც
  Walmart-ის (ან საერთოდ საცალო) მონაცემი არასდროს უნახავს. December gate: **REVIEW −13.8%** — მაგრამ
  შობის ცდომილება DLinear/ARIMA-ზე (−24%) **ბევრად ნაკლებია**, ე.ი. pretrained prior შობას უკეთ
  ალაგებს. Registry version 6, alias `timesfm`.
* ეს არის თანამედროვე დასკვნა: foundation მოდელი zero-shot-ად აჯობა კლასიკურ და მარტივ DL მოდელებს.

### 7.6. Deep Learning — N-BEATS

* **არქიტექტურა:** N-BEATS (Oreshkin et al., 2020) — fully-connected ბლოკების დასტა; თითო ბლოკი
  აგენერირებს **backcast**-ს (გამოაკლდება input-ს, ამიტომ შემდეგი ბლოკები მოდელავენ იმას, რაც წინამ
  ვერ დაიჭირა) და **forecast**-ს (ჯამდება საბოლოო პროგნოზში). DLinear-ის ორი წრფივი ფენისგან
  განსხვავებით ეს ღრმა არაწრფივი მოდელია — კითხვა: ხომ არ ყიდულობს ეს დამატებითი ტევადობა რამეს
  DLinear-ის (1816) მიღმა. იყენებს იმავე `WalmartPanel`-ს და `train_torch` loop-ს.
* **residual study (მთავარი შემოწმება):** DLinear-ის დასკვნის ზუსტი გამეორება — residual რეჟიმში
  **ნულოვანი ქსელი აღადგენს naive-ს** (0.0 სხვაობა). weight_decay-ის ზრდისას საუკეთესო კონფიგურაცია
  სწორედ ყველაზე მაღალი wd-ს (**1.0**) აღმოჩნდა. ე.ი. N-BEATS-ის დამატებითი ტევადობის მიუხედავად,
  წლიდან-წელს ნაშთი კვლავ ხმაურია გლობალური მოდელისთვის — capacity არ შველის.
* **შედეგი:** best (residual, L=52, width=256, 2 blocks, 4 layers, wd=1.0) recent **1808.1** ≈ naive
  (naive-ს ეთანაბრება, −0.05%). December gate: **REVIEW** (`xmas_aligned_lag` არ აქვს). Registry
  version 7, alias `nbeats`.

### 7.7. Deep Learning — PatchTST (transformer)

* **არქიტექტურა:** PatchTST (Nie et al., 2023) — lookback-ს ჭრის overlapping **patch**-ებად,
  თითოეულს წრფივად embed-ავს, Transformer encoder ურევს patch-ების მიმდევრობას, flatten head კი
  horizon-ზე ასახავს. ორი დიზაინის არჩევანი ზუსტად ერგება ამ მონაცემებს: **channel-independence**
  (ერთი საერთო მოდელი ყველა სერიაზე — `WalmartPanel`-ის კონტრაქტი) და **patching** (`patch_len=4,
  stride=4` — attention ხედავს თვე-მასშტაბის ნაწილებს, არა ცალკეულ ხმაურიან კვირას).
* **კითხვა:** ხომ არ პოულობს attention lookback-ზე სტრუქტურას, რომელსაც წრფივმა/MLP მოდელებმა ვერ.
  **პასუხი — არა.** default კონფიგურაცია 2347-ს იძლეოდა; tuning-მა 1861-მდე ჩამოიყვანა (დიდი ძებნის
  გაუმჯობესება), მაგრამ **naive-ს მაინც ვერ ჯობნის** (−3.0%). training loss ადრევე გავიდა plateau-ზე
  (naive-ის დონეზე).
* **შედეგი:** best (residual, L=52, patch4/stride4, d_model=128, 3 layers, wd=0.1) recent **1861.3**
  — **სამივე DL მოდელიდან ყველაზე სუსტი**. December gate: **REVIEW −23.4%** (DLinear-ის მსგავსი).
  Registry version 8, alias `patchtst`.
* დასკვნა: transformer-ის დამატებული მექანიზმი აქ **უარესია** მარტივ DLinear-ზე — 3,331 მოკლე სერია
  და ხმაურიანი წლიური ნაშთი აშიმშილებს მას.

### 7.8. Classical — Prophet

* **მიდგომა:** Prophet (Taylor & Letham, 2018) — **per-series** დეკომპოზიციური მოდელი: piecewise-linear
  trend + Fourier წლიური სეზონურობა + holiday effects, MAP-ით. სხვა კუთხე დიზაინის სივრცეში: **ლოკალური**
  (თითო მოდელი თითო `(Store, Dept)`-ზე — 3,331 fit, საერთო ძალა სერიებს შორის არ არის). `joblib`-ით
  პარალელდება, `uncertainty_samples=0` ჩქარობს — სრული პანელი ~3.3 წუთი. Cold-start სერიები (< 30
  დაკვირვება) ეცემა seasonal naive-ზე `predict_frame`-ით.
* **მთავარი გადაწყვეტა — `pre_christmas` holiday term:** flagged შობის კვირის **წინა** კვირას
  ვამატებთ ცალკე holiday რეგრესორად, რადგან daily-profile deconvolution-მა აჩვენა, რომ დეკემბრის
  პიკი **2012-12-21-ზეა**, არა flagged 2012-12-28-ზე. ეს Prophet-ის პასუხია Christmas alignment-ზე.
* **შედეგი — მოულოდნელად ძლიერი:** recent **1748.4** (naive-ს ჯობნის **3.3%-ით**). ეს **ჯობნის**
  ARIMA-ს (1791.7), DLinear-ს (1816.2), N-BEATS-ს (1808.1) და PatchTST-ს (1861.3) — ე.ი. **მეორე
  საუკეთესო არა-tree მოდელია** TimesFM-ის შემდეგ. December gate: **PASS** — შობის ცდომილება მხოლოდ
  **−1.5%** (predicted peak 2012-12-21 სწორად). **ერთადერთი არა-tree მოდელი, რომელიც gate-ს გადის**
  (boosting-ის წყვილის გარდა). `pre_christmas` term-მა იმუშავა. Registry version 9, alias `prophet`.

---

## 8. შედეგების საბოლოო შედარება

ცხრილში — WMAE (რაც ნაკლებია, მით უკეთესი). Gate = December gate-ის შედეგი.

| # | მოდელი | ოჯახი | mirror | recent | early | mean | Gate | Registry |
|---|---|---|---|---|---|---|---|---|
| 1 | **XGBoost (საუკეთესო)** | Tree | **1817.7** | **1616.7** | **1851.2** | **1761.9** | PASS | `@xgboost` v3 |
| 2 | LightGBM | Tree | 1864.2 | 1650.3 | 1910.0 | 1808.2 | PASS | v2 |
| 3 | TimesFM (zero-shot) | Foundation | — | 1679.1 | — | — | REVIEW −13.8% | `@timesfm` v6 |
| 4 | **Prophet** | Classical | — | **1748.4** | — | — | **PASS** | `@prophet` v9 |
| 5 | ARIMA | Classical | — | 1791.7 | — | — | REVIEW −23.9% | `@arima` v5 |
| — | *seasonal naive (baseline)* | — | 2037.8 | 1807.2 | 2018.6 | 1954.5 | — | — |
| 6 | N-BEATS | Deep Learning | — | 1808.1 | — | — | REVIEW | `@nbeats` v7 |
| 7 | DLinear | Deep Learning | — | 1816.2 | — | — | REVIEW −24% | `@dlinear` v4 |
| 8 | PatchTST | Deep Learning | — | 1861.3 | — | — | REVIEW −23.4% | `@patchtst` v8 |

*(DL/კლასიკური/foundation მოდელები `recent` fold-ზე ფასდებიან, რადგან ეს ერთადერთი fold-ია საკმარისი
ისტორიით 52-კვირიანი lookback-ისთვის.)*

### Kaggle-ის რეალური ლიდერბორდის ქულა

საუკეთესო მოდელის (**XGBoost**) `submissions/final_submission.csv` ატვირთულია Kaggle-ზე:

| | Public WMAE | Private WMAE |
|---|---|---|
| **XGBoost** | **2356.02** | **2423.48** |

ეს არის **მაღალი შედეგი** — competition-ის გამარჯვებულის private ქულა იყო ~2301,
ხოლო ძლიერი გადაწყვეტების უმეტესობა 2400–2900 დიაპაზონში იყო.

**რატომ არის Kaggle-ის ქულა (2356) უფრო მაღალი, ვიდრე CV `recent` (1616.7)?** ეს მოსალოდნელი და
ლოგიკურია: `recent` fold **არ შეიცავს შობას** ვალიდაციაში, რეალური test კი შეიცავს — და სწორედ
დეკემბრის კვირები (Christmas alignment, წონა-5) არის ამოცანის ყველაზე რთული ნაწილი. ე.ი. Kaggle-ის
ქულა უფრო „მკაცრ" პერიოდს ზომავს. სწორედ ამიტომ იყო კრიტიკული `xmas_aligned_lag` და December gate —
ისინი პირდაპირ ამ რთულ დეკემბრის კვირებზე მუშაობენ, რომელთაც CV ვერ ხედავს.

---

## 9. საუკეთესო მოდელი და Model Registry

ყველა მოდელის საუკეთესო ვარიანტი შენახულია **Pipeline-ად** (feature builder + მოდელი) და
დარეგისტრირებულია ერთი registered model-ის — `WalmartSalesForecast` — ვერსიებად, alias-ებით
(`xgboost`, `lightgbm`, `dlinear`, `arima`, `timesfm`, `nbeats`, `patchtst`, `prophet`). Pipeline
პირდაპირ დაუმუშავებელ `test.csv`-ზე ეშვება.

`model_inference.ipynb` აკეთებს შემდეგს (ტრენინგის გარეშე):

1. აგებს **leaderboard-ს** registry-დან — თითოეული ვერსიის CV WMAE-ს კითხულობს;
2. ირჩევს **საუკეთესოს** (ყველაზე დაბალი `recent` WMAE) → **XGBoost**;
3. ტვირთავს პირდაპირ registry-დან: `mlflow.pyfunc.load_model("models:/WalmartSalesForecast@xgboost")`;
4. აკეთებს predict-ს raw `test.csv`-ზე და აგენერირებს `submissions/final_submission.csv`-ს
   (Id-ს ამოწმებს `sampleSubmission.csv`-სთან);
5. ჩემპიონს ანიჭებს `champion` alias-ს.

**რატომ XGBoost:** ყველაზე დაბალი WMAE სამივე fold-ზე, საშუალო 1761.9, და **ერთადერთი** (LightGBM-თან
ერთად), რომელიც December gate-ს გადის — ე.ი. სწორად ალაგებს წონა-5 შობის კვირას.

---

## 10. MLflow-ის სტრუქტურა

ექსპერიმენტები დალოგილია **DagsHub MLflow**-ზე (`smama23/MLFinalProject`). თითოეული არქიტექტურისთვის
ცალკე ექსპერიმენტია, შიგნით run-ებით ეტაპების მიხედვით:

```
XGBoost_Training
├── XGBoost_Cleaning            # preprocessing + data-quality მეტრიკები
├── XGBoost_Baseline           # seasonal naive თითო fold-ზე
├── XGBoost_CV                  # wmae_{mirror,recent,early} + per-holiday MAE
├── XGBoost_Feature_Selection   # ablation-ები (nested run-ები)
├── XGBoost_Tuning              # random search (nested trial-ები)
└── XGBoost_Final              # Pipeline + December gate + registry
```

ანალოგიური სტრუქტურა: `LightGBM_Training`, `DLinear_Training`, `ARIMA_Training`, `TimesFM_Training`,
`NBEATS_Training`, `PatchTST_Training`, `Prophet_Training`. ყოველ run-ზე ვლოგავთ
`wmae_mirror/recent/early`-ს და per-holiday MAE-ს — ერთი WMAE რიცხვი მალავს იმას, დაეხმარა თუ
დააზიანა ცვლილება იმ 29.6% წონას, რომელიც სამ სადღესასწაულო კვირაზეა.

---

## 11. ძირითადი დასკვნები

1. **ხეები (XGBoost, LightGBM) იმარჯვებენ.** ისინი `lag_52`-ს **feature-ად** იყენებენ store/dept
   იდენტობასთან, `xmas_aligned_lag`-თან და ეგზოგენურ ცვლადებთან ერთად, და ლოკალურად split-ავენ. ეს
   უფრო ეფექტურია, ვიდრე დროის ფორმის გლობალურად მოდელირება.

2. **seasonal naive ძალიან ძლიერი baseline-ია.** ყველა გლობალური sequence მოდელი მას ძლივს ჯობნის ან
   უტოლდება: DLinear (1816), N-BEATS (1808), PatchTST (1861 — naive-ზე უარესი). DLinear-ის residual
   study და **N-BEATS-ის იმავე study-ის გამეორება** ერთ დასკვნაზე დგება: წლიდან-წელს ნაშთი გლობალური
   მოდელისთვის ხმაურია, და საუკეთესო რაც მას შეუძლია — naive-ის აღდგენა. N-BEATS-ის ღრმა ტევადობა და
   PatchTST-ის attention **არ შველის** — არქიტექტურის სირთულე აქ არ ყიდულობს სიგნალს.

3. **Foundation მოდელი (TimesFM) მოულოდნელად ძლიერია.** zero-shot-ად, ტრენინგის გარეშე, აჯობა
   კლასიკურ ARIMA-ს და ყველა DL მოდელს (1679 vs 1791/1808/1816/1861). ეს აჩვენებს pretrained
   ტემპორალური prior-ის ძალას.

4. **Christmas alignment არის ამოცანის გადამწყვეტი სირთულე**, და ის წონა-5 კვირაზეა. მოდელები,
   რომლებსაც `xmas_aligned_lag` არ აქვთ (DLinear, N-BEATS, PatchTST, ARIMA), December gate-ს ვერ
   გადიან (−23…−24%). TimesFM შუალედურია (−13.8%). **გამონაკლისი: Prophet გადის gate-ს (−1.5%)** —
   მისი explicit `pre_christmas` holiday term პირდაპირ ალაგებს დეკემბრის პიკს, ე.ი. per-series
   ინტერპრეტირებადი holiday რეგრესორი ცვლის `xmas_aligned_lag`-ს. ხეების გარდა ეს ერთადერთი მოდელია,
   რომელიც gate-ს გადის.

5. **Prophet არის საუკეთესო არა-tree, არა-foundation მოდელი.** recent 1748.4 — ჯობნის ARIMA-ს,
   DLinear-ს, N-BEATS-ს და PatchTST-ს, და ერთადერთია მათგან, რომელიც gate-ს გადის. per-series ლოკალური
   მოდელი explicit holiday term-ით აქ აჯობა გლობალურ sequence მოდელებს.

6. **SARIMA თეორიულად სწორია, პრაქტიკულად უვარგისი** ამ მონაცემებზე: 2.75 ციკლი სეზონურ პარამეტრებს
   იდენტიფიცირებადს ვერ ხდის (CI სიგანე ~54), და სრული პანელი ~6 საათი დასჭირდებოდა.

---

## 12. გუნდის განაწილება

პროექტი გუნდურია. ფუნდამენტური სამუშაოები (EDA, preprocessing, `src/`, ვალიდაცია, December gate) შესრულდა
ერთად. მოდელები განაწილდა:

### წევრი A (ეს ნაწილი — შესრულებული)
- ✅ `model_experiment_LightGBM.ipynb`
- ✅ `model_experiment_XGBoost.ipynb` (ჩემპიონი)
- ✅ `model_experiment_DLinear.ipynb` (+ `walmart_panel.py`, `walmart_dl.py`)
- ✅ `model_experiment_ARIMA_SARIMA.ipynb`
- ✅ `model_experiment_TimesFM.ipynb` (ბონუსი)
- ✅ `model_inference.ipynb` + Model Registry

### წევრი B (Deep Learning + Classical — შესრულებული)
- ✅ `model_experiment_NBEATS.ipynb` (+ `PatchTST` არქიტექტურა და `SeqForecaster` `walmart_dl.py`-ში)
- ✅ `model_experiment_PatchTST.ipynb` (transformer; PatchTST/TFT-იდან ერთი საკმარისია)
- ✅ `model_experiment_Prophet.ipynb`
- ✅ `model_inferenceV1.ipynb` გაშვება — leaderboard-ის განახლება 8 მოდელით, champion-ის ხელახალი არჩევა

---

## 13. გაშვების ინსტრუქცია

**გარემო:** Google Colab. DL მოდელები (DLinear, N-BEATS, PatchTST) და TimesFM საჭიროებენ **GPU
runtime**-ს (T4); Prophet, ARIMA და tree-მოდელები **CPU runtime**-ზე ეშვება. მონაცემები ინახება
Google Drive-ზე; ექსპერიმენტები ილოგება DagsHub MLflow-ზე.

1. მოათავსე რეპოზიტორია (`src/` და CSV-ები) Drive-ზე, ან clone-ი GitHub-იდან.
2. თითო `model_experiment_*.ipynb`-ს გაუშვი ზემოდან ქვემოთ. setup უჯრები ამონტაჟებენ Drive-ს,
   პოულობენ პროექტს და აკავშირებენ MLflow-ს (`dagshub.init`). DL/foundation notebook-ებს დაუყენე
   GPU runtime, Prophet-ს — CPU.
3. `model_inferenceV1.ipynb` ტვირთავს ჩემპიონს registry-დან და აგენერირებს
   `submissions/final_submission.csv`-ს Kaggle-ზე ასატვირთად.

**საჭირო ბიბლიოთეკები:** `lightgbm`, `xgboost`, `torch`, `statsmodels`, `timesfm[torch]`, `prophet`,
`joblib`, `mlflow`, `dagshub`, `scikit-learn`, `pandas`, `numpy`.

---

*დეტალური ტექნიკური დასაბუთება (ინგლისურად): [`docs/preprocessing_plan.md`](docs/preprocessing_plan.md).*
