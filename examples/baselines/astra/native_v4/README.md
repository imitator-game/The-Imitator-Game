Native success stops API calls immediately. No model finish tool exists.
Task success is native success_once, independent of endpoint coverage or timeout bookkeeping.
After native success, the closed-model host holds the last target to step 500 for a separate endpoint metric.
The 1800-second budget accumulates client LLM/agent planning time, including API waits/retries.
Initialization, simulation/rendering, host tools, post-success hold and artifact work do not consume it.
