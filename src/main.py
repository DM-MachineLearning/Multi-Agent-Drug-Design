import os
from madm.generators.multi_agent import MultiAgentPipeline

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # ensure no tokenizer warnings
    
    pipeline = MultiAgentPipeline()
    try:
        pipeline.run_optimization_loop(rounds=10, batch_size=16)
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        pipeline.save_final_models()