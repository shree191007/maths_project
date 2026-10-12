# Who Did What

## The project in 30 seconds

We took NASA's temperature records (1880 to 2025) and used maths to answer two questions:

1. **Can we fill in missing temperatures?** (called *reconstruction*)
2. **Can we guess future temperatures?** (called *extrapolation*)

These are different jobs. Reconstruction guesses a value **inside** the years we have. Extrapolation guesses a value **outside** them, in the future. Guessing inside is much easier.

We split the work into 4 parts. Each person owns one part, and **each part is one Python file**.

| | Person | Their job in one line | Their file |
|---|---|---|---|
| 1 | **Mathematics** | Fill in missing years, and find the main patterns | `person1_math.py` |
| 2 | **Data** | Get the NASA data and clean it | `person2_data.py` |
| 3 | **Regression** | Draw a trend line and use it to guess 2030 and 2040 | `person3_regression.py` |
| 4 | **Validation + Dashboard** | Test whether everything works, and put it all in one app | `person4_validation.py` |

```
Person 2 (data)  →  Person 1 (math)  ┐
                 →  Person 3 (trend) ├→  Person 4 (testing + dashboard)
```

Person 2 goes first because everyone else needs the data. Person 2's file also holds the chart colours and the "save a graph" helpers that the other three files use.

---

## Person 2: Data

**Job:** Get the temperature data and make it clean and ready to use.

**What I did:**
1. Downloaded the temperature file from NASA (it is called GISTEMP).
2. Loaded it into Python with pandas.
3. Cleaned it: NASA writes `***` where a value is missing, so I turned those into proper "empty" values. I also removed repeated years and put the years in order.
4. Checked the file: 146 years (1880 to 2025), no repeated years, no missing values.
5. Plotted the temperatures over time.
6. Turned the data into a **grid (matrix)**: 8 rows (regions of the Earth, from the North Pole down to the South Pole) and 146 columns (one for each year).
7. Saved the clean data as `data/processed/climate.csv`.

**What is an "anomaly"?** NASA doesn't give the actual temperature. It gives how much warmer or colder a year was than the 1951–1980 average. An anomaly of 0 means "normal". An anomaly of +1.2 means 1.2 °C warmer than normal.

**Result:** The graphs show the whole Earth warming up, and the Arctic (top row) warming the most.

**Something to know:** NASA's file has no missing values. So when we test our methods, we **hide some values on purpose**, then see whether the methods can guess them back.

**One sentence to say:** *"I collected and cleaned the climate data and got it ready for the maths."*

---

## Person 1: Mathematics

**Job:** Two things: (a) guess missing years, (b) find the main patterns in the data grid.

**What I did, part (a), guessing missing years (interpolation):**
- **Linear interpolation:** draw a straight line between the year before and the year after the gap. Read the missing value off the line.
- **Lagrange interpolation:** draw a smooth curve through several nearby years instead of a straight line.
- Then I hid some years and checked which method guessed them best.

**Result:** The simple straight line won. It was off by only about **0.09 °C**. The fancy curves did worse, because they copied the small random ups and downs in the data.

**What I did, part (b), finding patterns (SVD):**
- **SVD** is a way to break a big grid of numbers into a few simple "patterns", ranked from most important to least important.
- The most important pattern is "the whole Earth is getting warmer, the Arctic most of all". It explains **72%** of the ups and downs. The first two patterns together explain **92%**.
- **Rank** means "how many patterns you keep". If we rebuild the grid using only the top 2 patterns, we get very close to the original (off by about 0.17 °C on average). With 3 patterns, it is even closer (0.11 °C).
- **Reconstruction error** means "how far the rebuilt grid is from the real one". Fewer patterns means more error.

**One sentence to say:** *"I used interpolation to guess missing values and SVD to find the main patterns in the climate data."*

---

## Person 3: Regression

**Job:** Draw the best trend line through the data, and use it to guess future temperatures.

**What I did:**
- **Regression** means drawing the line (or curve) that fits all the points as well as it can overall. It doesn't pass through every point. It goes through the middle of them.
- **Least squares** is how we pick the "best" line. For each point we measure how far the line misses it, square that distance, and add them all up. The best line makes that total as small as possible.
- I tried two shapes: a **straight line** (linear) and a **curve** (quadratic).
- I used each to guess 2030 and 2040.
- I solved the maths in a careful way (called QR). The formula in the textbook goes wrong with big numbers like the year 2020, so I rescaled the years first.

**Result:**

| Model | 2030 | 2040 |
|---|---|---|
| Straight line | 0.72 °C | 0.81 °C |
| Curve | 1.13 °C | 1.37 °C |

The curve guesses higher because warming has been **speeding up**, and a straight line can't see that.

**Important:** if we only train on recent years (from 1980), the straight line guesses 1.36 °C for 2040 instead of 0.81 °C. So the answer depends a lot on which years we use.

**Warning:** this is just extending a curve. It is not a real climate forecast.

**One sentence to say:** *"I fitted a line and a curve to the temperature history and used them to estimate future temperatures."*

---

## Person 4: Validation + Dashboard

**Job:** Check whether the methods really work, and put everything together in one app.

**What I did:**
- I used two scores for "how wrong was the guess":
  - **MAE:** the average size of the mistakes.
  - **RMSE:** like MAE, but big mistakes count extra.
  - Smaller is better for both.
- **The golden rule:** only judge a method on data it has never seen.
- **Test 1, missing years:** I hid some years and checked how well each method guessed them back. The straight line was best (about 0.09 °C off).
- **Test 2, the future:** I pretended it was 1919, trained on the past, and guessed the next 10 years. Then I moved forward a year and repeated this 970 times. The curve was best (about 0.17 °C off). The fancier curve (cubic) fits the past best but guesses the future *worse*.
- **Test 3, the grid:** I hid values in the grid and rebuilt them using the patterns. Using 1 pattern helped. Using 3 or more patterns made it worse, because the method started copying random noise.
- **Test 4, different training years:** I changed which years the models learn from. The 2040 guess moved by up to about 0.5 °C.
- I built a **dashboard** with Streamlit (`streamlit run person4_validation.py`). It has 5 tabs, and the sliders re-run the real code.

**Result:** Guessing missing years is accurate to about **0.1 °C**. Guessing the future is less accurate, about **0.17 °C** at best. Our "95% sure" ranges were only right **68–86%** of the time, so they were too confident.

**Limits:** the NASA data has no real gaps, so all our gaps are artificial. We only have 8 regions, which is a small grid. And future guesses are curve-fitting, not physics.

**One sentence to say:** *"I tested all the methods with error scores and put the results in one interactive dashboard."*

---

## How it all fits together

1. **Person 2** gets the clean data and builds the grid.
2. **Person 1** uses it to fill gaps and find patterns.
3. **Person 3** uses it to draw trend lines and guess the future.
4. **Person 4** tests everything and shows it all in the dashboard.

## The answer to our question

- **Filling in missing years works well.** We were off by about 0.1 °C.
- **Two main patterns explain 92% of the variation** between regions. The biggest is "everywhere is warming, the Arctic most".
- **Guessing the future is much less reliable.** It is less accurate, our ranges were too confident, and the answer changes depending on which years we train on.

Being good at the first job does not mean being good at the second.

## Small word list

| Word | What it means |
|---|---|
| Anomaly | How much warmer or colder than the 1951–1980 average |
| Matrix | A grid of numbers (here: regions down the side, years across the top) |
| Interpolation | Guessing a value **between** known values |
| Extrapolation | Guessing a value **beyond** the known values (the future) |
| SVD | Breaking a grid into a few ranked patterns |
| Rank | How many patterns you keep |
| Regression | Fitting a line or curve through data |
| Least squares | The "best fit" rule: make the squared misses as small as possible |
| MAE / RMSE | Scores for how wrong the guesses are (smaller is better) |
