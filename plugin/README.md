# LoopLab generator

Select models, run automatic compatibility/quality checks, and chat through the local LoopLab companion from LM Studio.

Build and setup instructions are in the [project README](../README.md). The companion and patched runtime must be installed separately. The plugin does not modify LM Studio's bundled runtime.

In LM Studio, open **Chat → model picker**, search **looplab**, and select **bouldinnathan/looplab** under **Your Generators**. Enter the underlying model name in LoopLab's settings. The Developer/server picker and My Models do not list this generator; installing it does not create extra GGUF model entries.

Actions: Chat, Find models, Auto-test selected model, Auto-test all local models, Show report, Show test progress, Cancel active tests.

License: AGPL-3.0-or-later. See [LICENSE](../LICENSE).
