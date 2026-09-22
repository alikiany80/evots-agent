# EvoTS-Agent با OpenRouter

پیاده‌سازی مستقل پژوهشی برای آزمایش و توسعهٔ ایدهٔ مقالهٔ [EvoTS-Agent](https://arxiv.org/html/2608.17933v1).
این پروژه کد رسمی نویسندگان نیست و اعداد جداول مقاله را بازتولیدشده اعلام نمی‌کند.

دو حالت اجرایی وجود دارد:

| حالت | کار مدل زبانی | اجرا |
|---|---|---|
| `spec` | انتخاب مدل، نمایش داده، تنظیمات و ترکیب آشکارسازها | کد ازپیش‌نوشته‌شدهٔ بانک مدل‌ها در پردازش جدا |
| `python` | تولید و اصلاح تابع کامل تشخیص تغییر | کانتینر Docker بدون شبکه و بدون کلید API |

برای بررسی رفتار چرخه و هزینهٔ کمتر، از `spec` شروع کنید. برای توسعهٔ الگوریتم و تغییر آزاد کد، `python` را به کار ببرید. حالت `spec` فضای جست‌وجو را محدود می‌کند؛ آزمایش آن را معادل تولید آزاد کد در مقاله گزارش نکنید.

## ۱. نصب

Python 3.11 یا جدیدتر لازم است. از داخل پوشهٔ پروژه:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[forest,dev]"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m evots doctor
python -m pytest -q
```

در PowerShell، فعال‌سازی محیط با `.venv\Scripts\Activate.ps1` است. برای تست اولیهٔ سبک، `pip install -e ".[dev]"` کافی است؛ تنظیمات `smoke.json` وابستگی‌های اختیاری را به کار نمی‌گیرد.

## ۲. یک اجرای کامل بدون هزینهٔ API

```bash
python -m evots generate --kind mean_variance --n 3000 --count 3 --seed 42 --output data/mv
python -m evots prepare --data data/mv --output prepared/mv
python -m evots run --prepared prepared/mv --config configs/smoke.json --output runs/smoke
python -m evots evaluate --run runs/smoke
python -m evots report --run runs/smoke
```

فایل `runs/smoke/report.html` را باز کنید. این اجرا از پیشنهاددهندهٔ ساختگی و قطعی استفاده می‌کند؛ امتیازهایش فقط برای بررسی زیرساخت هستند و شاهدی برای عملکرد مدل زبانی یا مقاله محسوب نمی‌شوند. نمونهٔ اجراشده در `examples/smoke-run/` موجود است.

## ۳. اجرای واقعی با OpenRouter

کلید را فقط در محیط محلی خود تنظیم کنید:

```bash
export OPENROUTER_API_KEY="YOUR_KEY"
export OPENROUTER_MODEL="openai/gpt-4o"
python -m evots doctor --list-models
python -m evots run --prepared prepared/mv --config configs/openrouter.json --output runs/mv-real
python -m evots evaluate --run runs/mv-real
python -m evots report --run runs/mv-real
```

در PowerShell از `$env:OPENROUTER_API_KEY="YOUR_KEY"` و `$env:OPENROUTER_MODEL="openai/gpt-4o"` استفاده کنید. فایل `.env.example` فقط الگوست؛ برنامه `.env` را خودکار بارگذاری نمی‌کند.

`OPENROUTER_MODEL` بر نام مدل داخل تنظیمات اولویت دارد. مقدار نمونه توصیهٔ عملکردی نیست؛ مدل دلخواهتان را با شناسهٔ معتبر OpenRouter جایگزین کنید. `doctor --list-models` فهرست قابلیت‌های مدل انتخابی را بررسی می‌کند و درخواست تولید متن نمی‌فرستد.

در تنظیمات پیش‌فرض، `K=3` و تعداد مراحل پس از گرم‌کردن `8` است: در مجموع `2×3+8=14` آزمایش. یک درخواست انتخاب مدل هم وجود دارد و تلاش مجدد/ترمیم می‌تواند تعداد درخواست‌های API را بیشتر کند. این اعداد انتخاب این پروژه‌اند؛ مقادیر متناظر مقاله مشخص نشده‌اند.

### بودجه و تکرارپذیری

- `max_requests` سقف تعداد تلاش‌های شبکه، شامل retry، است.
- `max_cost_usd` بر هزینهٔ گزارش‌شدهٔ پاسخ‌های قبلی نظارت دارد. سقف قطعی صورتحساب نیست: یک درخواست می‌تواند از آن عبور کند و هزینهٔ timeout ممکن است نامعلوم باشد. محدودیت واقعی هزینه را روی کلید OpenRouter هم تنظیم کنید.
- پاسخ‌ها، درخواست‌ها، مصرف توکن و هزینهٔ گزارش‌شده داخل `llm/` ثبت می‌شوند؛ کلید API در آن‌ها نوشته نمی‌شود.
- با همان پوشه و `--resume` اجرا را ادامه دهید. ورودی‌ها، کد و تنظیمات علمی باید یکسان باشند؛ سقف درخواست/هزینه و تنظیمات retry را می‌توان تغییر داد.
- seed مربوط به داده، آشکارساز و نمونه‌گیری است. پاسخ API لزوماً قطعی نیست. `send_seed` فقط برای مدل/ارائه‌دهندهٔ پشتیبان فعال شود.
- برای تثبیت ارائه‌دهنده، `llm.provider.order` را با نام معتبر ارائه‌دهنده پر کنید. خاموش بودن fallback به‌تنهایی ارائه‌دهنده را ثابت نمی‌کند. شناسهٔ واقعی مدل/ارائه‌دهنده در پاسخ خام ذخیره می‌شود.

## ۴. تکامل آزاد کد

Docker باید نصب و در حال اجرا باشد:

```bash
docker build -t evots-worker:local .
python -m evots run --prepared prepared/mv --config configs/code_evolution.json --output runs/mv-code
python -m evots evaluate --run runs/mv-code
python -m evots report --run runs/mv-code
```

تابع تولیدشده این قرارداد را دارد:

```python
from evots.bank import detect as bank_detect

def detect(train, x, seed):
    # train, x: arrays of shape [time, features]
    # Return 0-based indices of the FIRST sample in each new segment.
    return bank_detect(train, x, {"model": "pelt", "penalty": 8.0}, seed)
```

کد تولیدی فقط آرایه‌های آموزش و دادهٔ بدون برچسبِ بخش مورد ارزیابی را می‌بیند. امتیاز بیرون از کانتینر محاسبه می‌شود. شبکه، کلید API، برچسب‌های اعتبارسنجی/آزمون و فایل‌های پروژهٔ میزبان به کانتینر داده نمی‌شوند. محدودیت زمان، حافظه و CPU قابل تنظیم است. این حالت عمداً راه اجرای آزاد Python روی میزبان ندارد.

`vision=true` نمودار بدون برچسب آموزش را نیز به مدل می‌فرستد؛ برای مدل‌های بدون ورودی تصویر آن را خاموش کنید و این تفاوت را در گزارش آزمایش بنویسید. در مدل‌هایی که `temperature` یا JSON mode را پشتیبانی نمی‌کنند، تنظیمات متناظر را تغییر دهید. برای کنترل reasoning از `llm.reasoning` استفاده کنید؛ برای مثال `{"enabled":false}` فقط روی مسیر پشتیبانی‌شده. پارامترهای پشتیبانی‌نشده به‌صورت پنهانی حذف نمی‌شوند.

## ۵. داده‌ها

### دو مولد مصنوعی

```bash
python -m evots generate --kind ou --n 6000 --count 6 --seed 123 --segment-length 200 --output data/ou
python -m evots prepare --data data/ou --output prepared/ou
```

مولد OU از گذار دقیق گسسته با پارامترهای قطعه‌ای استفاده می‌کند. مولد دیگر تغییر هم‌زمان میانگین و واریانس گاوسی دارد. توزیع پارامترها، طول قطعه‌ها و اندازهٔ داده در این پروژه مشخص و قابل تغییر است؛ این‌ها داده‌های منتشرشدهٔ نویسندگان نیستند.

### Bee-Dance

```bash
python -m evots fetch-bee --output data/bee
python -m evots prepare --data data/bee --output prepared/bee
```

شش ضبط از مخزن عمومی KL-CPD با commit ثابت دریافت می‌شود. هر ضبط مستقل و به‌ترتیب زمانی تقسیم می‌شود؛ ضبط‌ها به هم چسبانده نمی‌شوند. این انتخاب و تبدیل برچسب‌ها در metadata ثبت می‌شود. برای فایل محلی سازگار نیز `import-bee --mat FILE.mat --output data/bee-one` موجود است.

### دادهٔ خودتان

CSV باید ستون‌های عددی ویژگی را داشته باشد. برچسب‌ها یک فایل JSON مانند `[120, 245, 380]` هستند:

```bash
python -m evots import-csv --csv prices.csv --columns return volume --labels boundaries.json --output data/custom
python -m evots prepare --data data/custom --output prepared/custom
```

اعداد برچسب، شمارهٔ نمونه از صفر و محل شروع قطعهٔ جدید هستند؛ `0` و طول سری، مرز داخلی نیستند. ستون زمان یا برچسب را در `--columns` نگذارید. داده‌های بدون برچسب اعتبارسنجی برای این روشِ انتخاب مدل کافی نیستند.

### ADIA

دادهٔ ADIA در پروژه بسته‌بندی نشده است. نسخهٔ در دسترس خود را به CSV و manifest تبدیل کنید؛ قالب در `examples/adia_manifest.example.json` است:

```bash
python -m evots stitch-adia --manifest adia_manifest.json --half-window 150 --windows-per-stream 6 --output data/adia
python -m evots prepare --data data/adia --output prepared/adia
```

این سازنده، پنجره‌های اطراف شکست را بر اساس تغییر میانگین و نسبت نوسان گروه‌بندی می‌کند. محل اتصال مصنوعی پنجره‌ها جدا ثبت و به‌عنوان مرز برچسب‌گذاری می‌شود. دستور دقیق نویسندگان منتشر نشده؛ این خروجی را «ADIA با ساخت مستقل» بنامید. برای مقایسهٔ دقیق با مقاله، داده و دستور اصلی آن‌ها لازم است.

## ۶. ارزیابی جامع و ablation

```bash
python -m evots matrix --prepared prepared/bee --config configs/openrouter.json --output runs/bee-matrix --seeds 11 22 33 --variants full no_alternative no_recombination revision_only no_eda --include-baselines
```

این فرمان اجرای واقعی و هزینه‌دار است؛ بودجهٔ API برای هر اجرای مستقل اعمال می‌شود. تعداد اجراها برابر تعداد مدل‌ها × حالت‌ها × seedهاست. ابتدا تمام جست‌وجوها پایان می‌یابد، سپس آزمون مدل‌های تثبیت‌شده اجرا می‌شود. شکست‌ها حذف نمی‌شوند و نرخ موفقیت گزارش می‌شود.

خروجی‌های `summary.csv` و `summary.json` شامل میانگین، انحراف معیار نمونه، نرخ موفقیت کل اجرا و F1 با احتساب شکست‌ها هستند. میانگین اصلی روی اجراهای موفق است؛ ستون `failure_adjusted_f1` شکست را صفر حساب می‌کند. Hausdorff نامتناهی با `null` و پرچم جدا ثبت می‌شود، نه با عدد دلخواه.

برای چند مدل، `OPENROUTER_MODEL` را unset کنید و شناسه‌های دلخواه را به `--models` بدهید. تمام مدل‌ها باید قرارداد پاسخ تنظیم‌شده را پشتیبانی کنند. برای تنظیمات ناسازگار بین خانواده‌ها، ماتریس‌های جدا با تنظیمات ثبت‌شده اجرا کنید.

```bash
python -m evots baseline --prepared prepared/bee --config configs/openrouter.json --method random_search --output runs/random
python -m evots evaluate --run runs/random
python -m evots compare --matrix runs/bee-matrix/matrix.json --model openai/gpt-4o --a full --b no_alternative --output runs/bee-matrix/paired.json
```

خط مبناهای موجود: هر آشکارساز با تنظیم ثابت و جست‌وجوی تصادفی با تعداد آزمایش برابر. این‌ها TS-Agent، DS-Agent یا ResearchAgent نیستند. مقایسهٔ عین جدول مقاله به نسخه و تنظیمات رسمی آن سه روش نیاز دارد. bootstrap جفت‌شده روی seedها اکتشافی است؛ سه اجرا برای نتیجه‌گیری قوی دربارهٔ بهبود کوچک کافی نیست.

## ۷. پروتکل و خروجی‌ها

- تقسیم زمانی پیش‌فرض ۶۰/۲۰/۲۰ داخل هر سری انجام می‌شود؛ برچسب آموزش حذف می‌شود.
- پیش‌پردازشِ نیازمند fit فقط روی آموزش fit می‌شود. تشخیص این پروژه **آفلاین** است؛ دیدن آینده داخل همان بخش مجاز است و ادعای تشخیص برخط ندارد.
- تطبیق مرزها یک‌به‌یک، با بیشترین تعداد تطبیق ممکن و تلورانس شامل دو سر ±۱۰ است.
- انتخاب بر اساس میانگین F1 اعتبارسنجی بین سری‌هاست؛ دادهٔ آزمون تا `evaluate` بارگذاری نمی‌شود.
- بهترین جواب فقط با بهبود **اکیداً مثبت** عوض می‌شود. بهبود کوچک می‌تواند پذیرفته شود و در عین حال علامت توقف پیشرفت بگیرد.
- `trajectories/`: کد، اختلاف کد، والدها، عملگر، امتیاز، تصمیم و لاگ هر آزمایش.
- `best.json` و `best.py`: جواب تثبیت‌شده. `test_results.json`: نتیجهٔ آزمون همان جواب.
- `manifest.json`: اثرانگشت ورودی، کد، تنظیمات و نسخهٔ وابستگی‌ها. `llm/`: درخواست‌ها، پاسخ‌ها و مصرف.

## ۸. نقطهٔ شروع توسعه

در `docs/EXTENDING.md` محل اضافه‌کردن مدل، عملگر و پروتکل جدید توضیح داده شده است. قبل از نسبت‌دادن نتیجه به مقاله، جدول `docs/FIDELITY.md` را بخوانید. گزارش آزمون این تحویل در `docs/VALIDATION.md` است.

هیچ کلید API در این بسته وجود ندارد. برای تحویل اولیه، درخواست واقعی تولید متن به OpenRouter اجرا نشده است؛ تست اتصال با پاسخ‌های شبیه‌سازی‌شده انجام شده است.
