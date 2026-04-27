import sys
import asyncio

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from paperqa.agents.search import get_directory_index

from paperqa_config import health_check, make_settings, setup_run_log

log_path = setup_run_log("build")
print(f"Logging to {log_path}")

health_check()
settings = make_settings(rebuild_index=True)


async def build_index_robust():
    """Build the index, tolerating per-file failures."""
    print("Building index...")
    try:
        index = await get_directory_index(settings=settings, build=True)
        print("\nIndex build complete.")
        return index
    except Exception as e:
        print(f"\nSome files failed to index, falling back to existing index: {e}")
        settings.agent.rebuild_index = False
        return await get_directory_index(settings=settings, build=False)


asyncio.run(build_index_robust())
print("\nBuild phase done. Ready for questions.")
