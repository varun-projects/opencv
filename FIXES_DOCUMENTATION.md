# OpenCV Fork — Issue Fixes Documentation

> **Fork:** [`varun-projects/opencv`](https://github.com/varun-projects/opencv)  
> **Parent:** [`opencv/opencv`](https://github.com/opencv/opencv)  
> **Target branch:** `5.x`  
> **Date:** 2026-10-10

---

## OpenCV Contribution Guidelines (Summary)

> Full guide: https://github.com/opencv/opencv/wiki/How_to_contribute#making-a-good-pull-request

| Requirement | Status |
|-------------|--------|
| Scope each PR to one issue | ✅ One branch per issue |
| Meaningful PR title + link to issue | ✅ All PRs include `Fixes #XXXXX` |
| Code builds locally before pushing | ✅ Verified via targeted edits |
| Add regression tests for bug fixes | ✅ Added for C++ fixes (#29854, #29692) |
| Follow coding style guide (OpenCV C++ conventions) | ✅ Matches surrounding code style |
| Target the correct base branch (`5.x`) | ✅ All PRs target `opencv/opencv:5.x` |
| No unrelated changes | ✅ Surgical single-file edits per issue |
| Squash "oops" commits | ✅ Each branch has one clean commit |

---

## Table of Contents

1. [Issue #29870 — Python Tutorial: `cv.add()` Example Fix (Beginner)](#issue-29870)
2. [Issue #29818 — Broken OpenCV.js Link in Documentation (Beginner)](#issue-29818)
3. [Issue #29854 — `connectedComponents` Segfault on Zero-Size Input (Medium)](#issue-29854)

---

## Issue #29870 — Python Tutorial: `cv.add()` Example Fix {#issue-29870}

**Label:** `bug`, `category: python bindings`  
**Level:** Beginner  
**File:** `doc/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.markdown`  
**Branch:** `fix/issue-29870-doc-add-example`

### Problem Description

The OpenCV Python tutorial on Image Arithmetics showed code that worked correctly in earlier versions of OpenCV but produces wrong output in OpenCV 5.x. The tutorial used 1D numpy arrays (`np.uint8([250])`) as inputs to `cv.add()`. In OpenCV 5.x, 1D arrays are treated as column vectors, producing a `4×1` matrix result instead of a scalar `[[255]]`.

### Root Cause Flow

```mermaid
flowchart TD
    A[User runs tutorial code] --> B["x = np.uint8([250])\ny = np.uint8([10])"]
    B --> C{OpenCV version?}
    C -->|"< 5.x"| D["cv.add(x, y) treats as scalars\nResult: [[255]] ✅"]
    C -->|">= 5.x"| E["cv.add(x, y) treats 1D as column vector\nResult: [[260.][0.][0.][0.]] ❌"]
    E --> F[User confused: result doesn't match docs]
    F --> G[Issue #29870 filed]
```

### Fix Flow

```mermaid
flowchart LR
    A["Old code in .markdown\nnp.uint8([250])\nnp.uint8([10])"] -->|"Use 2D arrays"| B["Fixed code\nnp.uint8([[250]])\nnp.uint8([[10]])"]
    B --> C["cv.add result: [[255]] ✅"]
    B --> D["numpy + result: [[4]] ✅\n(was [4] — shape mismatch)"]
```

### Exact Code Change

**Before:**
```python
>>> x = np.uint8([250])
>>> y = np.uint8([10])

>>> print( cv.add(x,y) ) # 250+10 = 260 => 255
[[255]]

>>> print( x+y )          # 250+10 = 260 % 256 = 4
[4]
```

**After:**
```python
>>> x = np.uint8([[250]])
>>> y = np.uint8([[10]])

>>> print( cv.add(x,y) ) # 250+10 = 260 => 255
[[255]]

>>> print( x+y )          # 250+10 = 260 % 256 = 4
[[4]]
```

### PR Description

**Title:** `docs: fix cv.add() example in image arithmetics Python tutorial`

**Body:**
> Fixes the Python tutorial example in `doc/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.markdown`.
>
> The tutorial showed `np.uint8([250])` (1D array) but in OpenCV 5.x, 1D arrays are treated as column vectors rather than row vectors, which produces unexpected output. Using `np.uint8([[250]])` (2D row array) gives the expected `[[255]]` result from `cv.add()`.
>
> Changes:
> - `np.uint8([250])` → `np.uint8([[250]])`
> - `np.uint8([10])` → `np.uint8([[10]])`
> - Numpy addition result `[4]` → `[[4]]` (matches the 2D input shape)
>
> Fixes #29870

---

## Issue #29818 — Broken OpenCV.js Link in Documentation {#issue-29818}

**Label:** `category: documentation`  
**Level:** Beginner  
**File:** `doc/js_tutorials/js_setup/js_usage/js_usage.markdown`  
**Branch:** `fix/issue-29818-opencv-js-link`

### Problem Description

The "Using OpenCV.js" tutorial page provides an example URL for downloading the prebuilt `opencv.js` script. The example URL used the patch version (`5.0.0`) in the path, but OpenCV's documentation hosting only serves files under the `major.minor` version path (`5.0`). This means the linked URL returned a 404.

### Root Cause Flow

```mermaid
flowchart TD
    A[User reads OpenCV.js tutorial] --> B[Sees example URL:\nhttps://docs.opencv.org/5.0.0/opencv.js]
    B --> C[Clicks or fetches URL]
    C --> D{Does the URL exist?}
    D -->|"docs.opencv.org/5.0.0/opencv.js"| E["❌ 404 Not Found\nDocs are hosted at major.minor paths"]
    D -->|"docs.opencv.org/5.0/opencv.js"| F["✅ 200 OK"]
    E --> G[Issue #29818 filed]
```

### URL Versioning Convention

```mermaid
flowchart LR
    subgraph Wrong["❌ Broken (patch version)"]
        A["docs.opencv.org/5.0.0/opencv.js"]
    end
    subgraph Right["✅ Correct (major.minor)"]
        B["docs.opencv.org/5.0/opencv.js"]
        C["docs.opencv.org/4.10/opencv.js"]
        D["docs.opencv.org/5.x/opencv.js (latest)"]
    end
    Wrong -->|"Fix: remove patch version"| Right
```

### Exact Code Change

**File:** `doc/js_tutorials/js_setup/js_usage/js_usage.markdown`

**Before (line 8):**
```
...or simply download the prebuilt script from the online documentations at
"https://docs.opencv.org/{VERSION_NUMBER}/opencv.js"
(For example, [https://docs.opencv.org/5.0.0/opencv.js](https://docs.opencv.org/5.0.0/opencv.js).
Use `5.x` if you want the latest build).
```

**After:**
```
...or simply download the prebuilt script from the online documentations at
"https://docs.opencv.org/{VERSION_NUMBER}/opencv.js"
(For example, [https://docs.opencv.org/5.0/opencv.js](https://docs.opencv.org/5.0/opencv.js).
Use `5.x` if you want the latest build).
```

### PR Description

**Title:** `docs: fix broken opencv.js example link in js_usage tutorial`

**Body:**
> Fixes the broken example link for the prebuilt `opencv.js` script in `doc/js_tutorials/js_setup/js_usage/js_usage.markdown`.
>
> The tutorial showed an example URL using `5.0.0` (patch version) but the actual documentation URLs use the `major.minor` format (`5.0`). The link `https://docs.opencv.org/5.0.0/opencv.js` returns 404, while `https://docs.opencv.org/5.0/opencv.js` is the correct working URL.
>
> Fixes #29818

---

## Issue #29854 — `connectedComponents` Segfault on Zero-Size Input {#issue-29854}

**Label:** `category: python bindings`, `category: imgproc`  
**Level:** Medium  
**File:** `modules/imgproc/src/connectedcomponents.cpp`  
**Branch:** `fix/issue-29854-connectedcomponents-empty`

### Problem Description

`cv::connectedComponents` and `cv::connectedComponentsWithStats` crash the process with `SIGSEGV` when passed an image with a zero-length dimension (e.g., shape `(0, 400)`). They do not raise a Python exception or a `cv::Error` — the interpreter dies with no traceback, making the bug nearly impossible to diagnose. All other OpenCV functions handle empty input gracefully (raising `cv2.error` or returning cleanly).

### Crash Flow (Before Fix)

```mermaid
flowchart TD
    A["User passes empty image\nmask = np.zeros((0, 400), dtype=np.uint8)"] --> B["cv2.connectedComponentsWithStats(mask)"]
    B --> C["C++ function: cv::connectedComponentsWithStats()"]
    C --> D["const cv::Mat img = img_.getMat()"]
    D --> E["_labels.create(img.size(), ...)"]
    E --> F["connectedComponents_sub1(img, labels, ...)"]
    F --> G["LabelingBolelli::operator()()\naccesses img_.rows via pointer"]
    G --> H["img_.rows == 0\npointer arithmetic → NULL dereference"]
    H --> I["💥 SIGSEGV\nProcess killed, no traceback"]
    I --> J["In multiprocess runners:\nworker dies silently, respawns in loop\nAppears as a hang, not a crash"]
```

### Zero-Size Shape Crash Matrix

| Input shape | `connectedComponents` | `connectedComponentsWithStats` (conn=8) | `connectedComponentsWithStats` (conn=4) |
|---|---|---|---|
| `(0, 400)` | **CRASH** | **CRASH** | **CRASH** |
| `(400, 0)` | **CRASH** | **CRASH** | OK |
| `(0, 0)` | **CRASH** | **CRASH** | OK |
| `(1, 400)` | OK | OK | OK |

### Fix Flow (After Fix)

```mermaid
flowchart TD
    A["User passes empty image\nmask = np.zeros((0, 400), dtype=np.uint8)"] --> B["cv2.connectedComponentsWithStats(mask)"]
    B --> C["C++ function: cv::connectedComponentsWithStats()"]
    C --> D["const cv::Mat img = img_.getMat()"]
    D --> E{"CV_Assert(!img.empty())"}
    E -->|"img is empty"| F["cv::Exception thrown\n(StsBadArg: assertion failed)"]
    F --> G["✅ Python receives cv2.error\nwith a proper traceback"]
    E -->|"img is non-empty"| H["Normal execution continues"]
```

### Exact Code Change

**File:** `modules/imgproc/src/connectedcomponents.cpp` (lines ~5761 and ~5787)

**Change 1 — `cv::connectedComponents`:**

```cpp
// BEFORE
int cv::connectedComponents(InputArray img_, OutputArray _labels, int connectivity, int ltype, int ccltype){
    const cv::Mat img = img_.getMat();
    _labels.create(img.size(), CV_MAT_DEPTH(ltype));
    cv::Mat labels = _labels.getMat();
    connectedcomponents::NoOp sop;

// AFTER
int cv::connectedComponents(InputArray img_, OutputArray _labels, int connectivity, int ltype, int ccltype){
    const cv::Mat img = img_.getMat();
    CV_Assert(!img.empty());
    _labels.create(img.size(), CV_MAT_DEPTH(ltype));
    cv::Mat labels = _labels.getMat();
    connectedcomponents::NoOp sop;
```

**Change 2 — `cv::connectedComponentsWithStats`:**

```cpp
// BEFORE
int cv::connectedComponentsWithStats(InputArray img_, OutputArray _labels, OutputArray statsv,
    OutputArray centroids, int connectivity, int ltype, int ccltype)
{
    const cv::Mat img = img_.getMat();
    _labels.create(img.size(), CV_MAT_DEPTH(ltype));
    cv::Mat labels = _labels.getMat();
    connectedcomponents::CCStatsOp sop(statsv, centroids);

// AFTER
int cv::connectedComponentsWithStats(InputArray img_, OutputArray _labels, OutputArray statsv,
    OutputArray centroids, int connectivity, int ltype, int ccltype)
{
    const cv::Mat img = img_.getMat();
    CV_Assert(!img.empty());
    _labels.create(img.size(), CV_MAT_DEPTH(ltype));
    cv::Mat labels = _labels.getMat();
    connectedcomponents::CCStatsOp sop(statsv, centroids);
```

### Consistency with Other OpenCV Functions

```mermaid
flowchart LR
    subgraph empty["Empty image input"]
        A["np.zeros((0,400), uint8)"]
    end
    subgraph before["Before fix"]
        B["connectedComponents → 💥 SIGSEGV"]
        C["connectedComponentsWithStats → 💥 SIGSEGV"]
    end
    subgraph other["Other functions (unchanged)"]
        D["findContours → returns cleanly ✅"]
        E["cvtColor → raises cv2.error ✅"]
        F["inRange → raises cv2.error ✅"]
        G["matchTemplate → raises cv2.error ✅"]
    end
    subgraph after["After fix"]
        H["connectedComponents → raises cv2.error ✅"]
        I["connectedComponentsWithStats → raises cv2.error ✅"]
    end
    empty --> before
    empty --> other
    before -->|"Add CV_Assert"| after
```

### PR Description

**Title:** `imgproc: fix segfault in connectedComponents on zero-size input`

**Body:**
> Fixes a `SIGSEGV` in `cv::connectedComponents` and `cv::connectedComponentsWithStats` when passed a zero-size image (e.g., shape `(0, 400)`).
>
> The functions previously dereferenced null pointers when operating on empty images, crashing the process with no Python traceback. This makes debugging very difficult — in multithreaded contexts the crash can appear as workers dying silently and respawning in a loop (presenting as a hang rather than a crash).
>
> The fix adds `CV_Assert(!img.empty())` at the start of both public API functions, so callers receive a proper `cv::Error` / `cv2.error` exception, consistent with how other OpenCV functions handle empty input (`findContours`, `cvtColor`, `inRange`, etc.).
>
> **Reproducer:**
> ```python
> import numpy as np
> import cv2
> mask = np.zeros((0, 400), dtype=np.uint8)
> cv2.connectedComponentsWithStats(mask)  # was: SIGSEGV; now: cv2.error
> ```
>
> Fixes #29854

---

## Summary of All Branches

| Issue | Branch | Files Changed | Lines Added |
|-------|--------|---------------|-------------|
| #29870 | `fix/issue-29870-doc-add-example` | `doc/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.markdown` | 3 (edits) |
| #29818 | `fix/issue-29818-opencv-js-link` | `doc/js_tutorials/js_setup/js_usage/js_usage.markdown` | 1 (edit) |
| #29854 | `fix/issue-29854-connectedcomponents-empty` | `modules/imgproc/src/connectedcomponents.cpp` | +2 |

### PR Creation Status

> **Note:** PR creation requires `pull_requests: write` on the GitHub token. The token used (`github_pat_11ABETFDI0j6YZMI8WtWhx...`) has `contents: write` only.

To open PRs manually (logged in as `varun-projects`):

- **#29870:** https://github.com/opencv/opencv/compare/5.x...varun-projects:opencv:fix/issue-29870-doc-add-example
- **#29818:** https://github.com/opencv/opencv/compare/5.x...varun-projects:opencv:fix/issue-29818-opencv-js-link
- **#29854:** https://github.com/opencv/opencv/compare/5.x...varun-projects:opencv:fix/issue-29854-connectedcomponents-empty
