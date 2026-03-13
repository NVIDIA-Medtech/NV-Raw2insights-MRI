# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os


def find_first_prefix_match(string, prefix_string):
    """
    Returns the index of the first string in 'strings' that starts with 'prefix'.
    If no match is found, returns -1.

    Args:
        prefix (str): The prefix to search for
        strings (list): List of strings to search in

    Returns:
        int: Index of the first match or -1 if not found
    """
    for i, prefix in enumerate(prefix_string):
        if string.startswith(prefix):
            return i
    return -1


def get_mask_files(source_dir, include_dir, mask_strs=["mask"], val_new_id=None):
    source_dir = os.path.abspath(source_dir)
    mask_files_found = 0
    mask_files = {}
    undermasklist_task1 = ["Uniform4", "Uniform8", "Uniform10"]
    undermasklist_task2 = ["ktUniform", "ktGaussian", "ktRadial"]

    if not isinstance(mask_strs, list):
        mask_strs = [mask_strs]

    for root, _, files in os.walk(source_dir):
        # Filter for .mat files
        mat_files = [f for f in files if f.endswith(".mat")]

        for mat_file in mat_files:

            if "MultiCoil" not in root or "AccFactor" in root:
                continue

            # Get the parent folder name
            if "Center" in root:  # if CMRxRecon 2025
                parent_folder = "_".join(root.split("/")[-3:])
            else:
                parent_folder = os.path.basename(root)

            # Create the new filename with parent folder as prefix
            new_filename = f"{parent_folder}_{mat_file}"

            if include_dir not in root:
                continue

            # Check if file should be excluded
            for mask_str in mask_strs:
                if mask_str in new_filename:
                    mask_files_found += 1
                    case_id = new_filename.split(f"_{mask_str}_")[0]
                    patient_id = case_id.split("_")[0]
                    try:
                        mask_type = new_filename.split(f"_{mask_str}_")[1].split(".mat")[0]
                    except BaseException:
                        mask_type = new_filename.split(f"_{mask_str}")[1].split(".mat")[0]

                    if "ValidationSet" in root and "2024" in root:
                        if "Mask_Task1" in root:
                            new_patient_id = val_new_id.loc[
                                val_new_id[f"file_info2{3 + find_first_prefix_match(mask_type, undermasklist_task1)}"]
                                == patient_id,
                                "file_info22",
                            ].item()
                        elif "Mask_Task2" in root:
                            new_patient_id = val_new_id.loc[
                                val_new_id[f"file_info2{3 + find_first_prefix_match(mask_type, undermasklist_task2)}"]
                                == patient_id,
                                "file_info22",
                            ].item()
                        else:
                            raise ValueError("Mask_Task1 and/or Mask_Task2 are not defined")
                        new_case_id = case_id.replace(patient_id, new_patient_id)
                    else:
                        new_case_id = case_id

                    # print(f"Found mask file: {os.path.join(root, mat_file)}")
                    if new_case_id not in mask_files:
                        mask_files[new_case_id] = []
                    mask_files[new_case_id].append(os.path.join(root, mat_file))
                else:
                    continue

    return mask_files, mask_files_found


def create_json_with_masks(
    source_dir,
    output_dir,
    mask_files,
    include_dir,
    mask_strings=None,
    exclude_files=None,
):
    # Convert to absolute paths
    source_dir = os.path.abspath(source_dir)
    output_dir = os.path.abspath(output_dir)

    # Set default exclude list if none provided
    if mask_strings is None:
        mask_strings = []
    if exclude_files is None:
        exclude_files = []

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Count for summary
    mat_files_found = 0
    json_files_created = 0
    files_excluded = 0

    # Walk through the directory tree
    for root, _, files in os.walk(source_dir):
        # Filter for .mat files
        mat_files = [f for f in files if f.endswith(".mat")]

        for mat_file in mat_files:
            mat_files_found += 1

            if "MultiCoil" not in root or "AccFactor" in root:
                files_excluded += 1
                continue

            # Get the parent folder name
            if "Center" in root:  # if CMRxRecon 2025
                parent_folder = "_".join(root.split("/")[-3:])
            else:
                parent_folder = os.path.basename(root)

            case_id = f"{parent_folder}_{mat_file.replace('.mat', '')}"

            # Create the new filename with parent folder as prefix
            new_filename = f"{case_id}.json"

            # Check if file should be excluded
            should_exclude = False
            for exclude_str in mask_strings:
                if exclude_str in mat_file:
                    should_exclude = True
                    files_excluded += 1
                    print(f"Excluding file (contains '{exclude_str}'): {os.path.join(root, mat_file)}")
                    break

            if include_dir not in root:
                should_exclude = True
                files_excluded += 1
                print(f"Excluding file (not in '{include_dir}'): {os.path.join(root, mat_file)}")

            for exclude_file in exclude_files:
                if exclude_file in new_filename.split(".mat")[0]:
                    should_exclude = True
                    files_excluded += 1
                    print(f"Excluding file (contains '{exclude_file}'): {os.path.join(root, mat_file)}")
                    break

            if should_exclude:
                continue

            # Full paths
            source_file = os.path.join(root, mat_file)
            output_file = os.path.join(output_dir, new_filename)

            # Create symbolic link
            try:
                # Check if link already exists and remove it
                if os.path.exists(output_file):
                    os.remove(output_file)
                if "kus" in case_id:
                    case_id = case_id.split("_kus")[0]
                mask = mask_files[case_id] if case_id in mask_files else []
                data_dict = {"kspace": source_file, "mask": mask}
                with open(output_file, "w") as f:
                    json.dump(data_dict, f, indent=4)
                print(f"Created json file: {output_file} with mask: {mask}")
                json_files_created += 1
            except Exception as e:
                print(f"Error creating json file for {source_file}: {e}")

    print("\nSummary:")
    print(f"- Found {mat_files_found} .mat files in {source_dir}")
    print(f"- Excluded {files_excluded} files based on exclusion patterns")
    print(f"- Created {json_files_created} json file in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create symbolic links for .mat files")
    parser.add_argument("--source_dir", help="Source directory to search for .mat files of kspace data")
    parser.add_argument(
        "--mask_dir",
        help="Mask directory to search for .mat files of mask data. Should always be path/to/ChallengeData",
    )
    parser.add_argument("--output_dir", help="Output directory for saved json files")
    parser.add_argument(
        "--training_set",
        action="store_true",
        default=False,
        help="Whether to create json files for training set. Default is False.",
    )
    args = parser.parse_args()

    if args.training_set:
        include_dirs = ["TrainingSet"]
    else:
        include_dirs = ["ValidationSet"]
    # TODO: add TestSet

    mask_strings = ["mask"]

    for include_dir in include_dirs:
        if include_dir == "TrainingSet" and "2024" in args.source_dir:
            exclude_files = [
                "P077_aorta_sag",
                "P105_tagging",
                "P065_aorta_sag",
                "P117_aorta_tra",
                "P087_tagging",
                "P105_aorta_tra",
                "P109_cine_sax",
                "P100_cine_sax",
            ]
        elif include_dir == "ValidationSet" and "2024" in args.source_dir:
            exclude_files = ["P040", "P018"]
        else:
            exclude_files = []
        mask_files, mask_files_found = get_mask_files(args.mask_dir, include_dir, mask_strings, val_new_id=None)
        create_json_with_masks(
            args.source_dir,
            args.output_dir,
            mask_files,
            include_dir,
            mask_strings,
            exclude_files,
        )

# 2025
# Usage Train :
# python scripts/create_cmrxrecon_dataset.py --source_dir /data/CMRxRecon2025_Data/ChallengeData --mask_dir \
# /data/CMRxRecon2025_Data/ChallengeData --output_dir dataset/CMRxRecon2025/ChallengeDataTrain --training_set
# Usage Val:
# python scripts/create_cmrxrecon_dataset.py --source_dir /data/CMRxRecon2025_Data/ChallengeDataValSet/TaskR1 \
# --mask_dir /data/CMRxRecon2025_Data/ChallengeDataValSet/TaskR1 --output_dir dataset/CMRxRecon2025/ChallengeDataValR1
