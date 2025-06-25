moe_layers="10"


# DynMoE (Ours)
python search_glue_no_trainer.py --model_name_or_path bert-large-cased --task_name cola --to_MoE --num_experts 8 --moe_layers $moe_layers --expert_repeat 16 --random_cluster --save_model --max_expert_num 16 --adaptive_experts --gate_type gated_multi_gate --save_model;


python test_glue_no_trainer.py --task_name cola_ood --use_fp16 --model_name_or_path bert-large-cased --source_dir /private/yjy/project/DILMoE/EMoE/Language/results/bert-large-cased_save/cola/learn_gate_random_False_repeat16/MoE_[10]_experts_8_top_k_4_key_gate_False_gated_multi_gate_aux_0.01_noise_1.0_capacity_1.5 --disable_peft --expert_repeat 16 --adaptive_experts --max_expert_num 16 --gpu 0;