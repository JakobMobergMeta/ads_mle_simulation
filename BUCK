load("@fbcode_macros//build_defs:python_library.bzl", "python_library")

oncall("ads_llm_productivity")

python_library(
    name = "bento_complexity_heatmap",
    srcs = ["bento_complexity_heatmap.py"],
    typing = True,
    deps = [
        "fbsource//third-party/pypi/matplotlib:matplotlib",
        "fbsource//third-party/pypi/numpy:numpy",
        ":model_performance_api",
    ],
)

python_library(
    name = "model_performance_api",
    srcs = ["model_performance_api.py"],
    typing = True,
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        ":simple_interpolation",
    ],
)

python_library(
    name = "simple_interpolation",
    srcs = ["simple_interpolation.py"],
    typing = True,
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/scipy:scipy",
    ],
)
