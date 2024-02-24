"""Profile the streamlit app to find bottlenecks."""

import cProfile
import pstats
import pyprojroot

import Home

# TODO Profile all pages: all at once, or take page name as argument?

def profile_streamlit_page():

    # Run the profiler
    profiler = cProfile.Profile()
    profiler.enable()
    Home.main()
    profiler.disable()

    # Writing the stats to a file
    output_file = pyprojroot.here("streamlit/logs/profile_stats.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as stream:
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        stats.print_stats()


if __name__ == "__main__":
    profile_streamlit_page()
