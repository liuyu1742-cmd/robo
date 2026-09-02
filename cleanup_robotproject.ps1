$ErrorActionPreference = 'Stop'
$root = (Resolve-Path 'C:\RobotProject\RobotProject').Path
$logPath = 'C:\RobotProject\cleanup_robotproject.log'
Set-Content -LiteralPath $logPath -Value "Started $(Get-Date -Format o)"
$drive = [System.IO.DriveInfo]::new('C')
Add-Content -LiteralPath $logPath -Value "AvailableFreeBytesBefore=$($drive.AvailableFreeSpace)"

$keepRoot = @('.git','.venv','datasets','docs','models','tests','tools','robocasa_gt_grasp.py','build_assistant_semantic_selftest.py','build_augmented_semantic_action_dataset.py','build_official_fixed_command_acceptance.py','build_semantic_action_dataset.py','evaluate_acceptance_instruction_benchmark.py','evaluate_assistant_semantic_selftest.py','evaluate_hybrid_semantic_development.py','evaluate_hybrid_semantic_final.py','evaluate_independent_blind_action_test.py','evaluate_instruction_benchmark.py','evaluate_project_actions.py','evaluate_project_model_v2.py','final_demo.py','manual_instruction_test.py','prepare_blind_paraphrase_test.py','probe_instructions.py','run_acceptance_check.py','run_delivery_check.py','run_ten_instruction_tests.py','train_instruction_action_transformer.py','train_project_transformer_v2.py','train_semantic_action_parser.py','requirements.txt')
$targets = [System.Collections.Generic.List[string]]::new()
Get-ChildItem -Force $root | Where-Object { $_.Name -notin $keepRoot } | ForEach-Object { [void]$targets.Add($_.FullName) }

$dataRoot = Join-Path $root 'datasets'
$keepDataDirs = @('epic_kitchens','midterm_15task_delivery','unified_actions','unified_actions_project')
Get-ChildItem -Force $dataRoot -Directory | Where-Object { $_.Name -notin $keepDataDirs } | ForEach-Object { [void]$targets.Add($_.FullName) }
$keepDataFilesPattern = '^(acceptance_|action_constraint|assistant_semantic|dataset_code_coverage|demo_cases|delivery_check|final_demo_cases|human_only|independent_blind|instruction_|manual_|official_|project_|semantic_|smoke_generation|smoke_manual|smoke_standard|standard_generation|midterm_15task_delivery_summary|midterm_command_runtime|object_operation_coverage|p1_evidence|public_annotation|vision_gap)'
Get-ChildItem -Force $dataRoot -File | Where-Object { $_.Name -notmatch $keepDataFilesPattern } | ForEach-Object { [void]$targets.Add($_.FullName) }

$modelsRoot = Join-Path $root 'models'
$keepModels = @('semantic_action','semantic_action_augmented','action_transformer_project_final.pt','action_transformer_project_generation_final.pt','action_transformer_project_generation_semantic.pt','action_transformer_project_v2.pt','action_transformer_project_visual.pt','instruction_action_transformer_v1.pt')
Get-ChildItem -Force $modelsRoot | Where-Object { $_.Name -notin $keepModels } | ForEach-Object { [void]$targets.Add($_.FullName) }

$testsRoot = Join-Path $root 'tests'
$keepTestPattern = '^(__init__\.py|test_(acceptance|action|assistant_semantic|blind|diverse_instruction|four_object_actions|household_action_plans|hybrid_semantic|independent_blind|instruction|manual_instruction|official|project|semantic|train_semantic))'
Get-ChildItem -Force $testsRoot | Where-Object { $_.Name -notmatch $keepTestPattern } | ForEach-Object { [void]$targets.Add($_.FullName) }

$toolsRoot = Join-Path $root 'tools'
$keepTools = @('__init__.py','action_constraints.py','blind_error_diagnostics.py','blind_paraphrase.py','build_acceptance_artifacts.py','build_demo_cases.py','build_final_dataset.py','build_project_action_datasets.py','build_project_condition_vocab.py','build_project_vocab.py','build_standard_generation_datasets.py','build_unified_action_sequences.py','checkpoint.py','check_project.py','convert_format.py','dataloader.py','dataset_transformer.py','dataset_validator.py','encode.py','execution.py','generation.py','household_catalog.py','household_dataset.py','hybrid_semantic_action_runtime.py','independent_blind_evaluation.py','instruction_action_model.py','instruction_benchmark.py','instruction_parser.py','manual_instruction_entry.py','model.py','official_acceptance_data.py','operation_plans.py','project_conditions.py','project_encode.py','project_generation.py','project_runtime.py','robot_transformer.py','semantic_action_augmentation.py','semantic_action_data.py','semantic_action_model.py','transformer_model.py','vla_action_templates.py')
Get-ChildItem -Force $toolsRoot | Where-Object { $_.Name -notin $keepTools } | ForEach-Object { [void]$targets.Add($_.FullName) }

$targets = $targets | Sort-Object -Unique
foreach ($path in $targets) {
    $resolved = (Resolve-Path -LiteralPath $path).Path
    if (-not $resolved.StartsWith($root + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe target outside project: $resolved"
    }
}

foreach ($path in $targets) {
    try {
        Remove-Item -LiteralPath $path -Force -Recurse -ErrorAction Stop
        Add-Content -LiteralPath $logPath -Value "DELETED $path"
    }
    catch {
        Add-Content -LiteralPath $logPath -Value "FAILED  $path :: $($_.Exception.Message)"
    }
}
Add-Content -LiteralPath $logPath -Value "Finished $(Get-Date -Format o)"
$drive = [System.IO.DriveInfo]::new('C')
Add-Content -LiteralPath $logPath -Value "AvailableFreeBytesAfter=$($drive.AvailableFreeSpace)"
