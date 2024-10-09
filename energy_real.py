import json
import os

import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.scoring import ScoreFunction

pyrosetta.init()


def save_json(data, file_path, indent=4, **kwargs):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf8") as f:
        f.write(f"{json.dumps(data, ensure_ascii=False, indent=indent, **kwargs)}\n")


def main(result_dir, save_dir):
    # initialize
    all_energy = []

    """每个PDB的各自能量"""
    for file_name in sorted(os.listdir(result_dir)):
        # file_name example: "gen_0.pdb"
        # 读取PDB
        pose = pose_from_pdb(os.path.join(result_dir, file_name))

        # 创建打分函数
        scorefxn = ScoreFunction()  # 创建一个空的打分函数

        # https://docs.rosettacommons.org/docs/latest/rosetta_basics/scoring/score-types
        # hbond_sr_bb: Backbone-backbone hbonds close in primary sequence.  All hydrogen bonding terms support canonical and noncanonical types.
        scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_sr_bb, 1.0)

        # hbond_lr_bb: Backbone-backbone hbonds distant in primary sequence.
        scorefxn.set_weight(pyrosetta.rosetta.core.scoring.hbond_lr_bb, 1.0)

        # rama: Ramachandran preferences.  Supports only the 20 canonical alpha-amino acids and their mirror images.
        scorefxn.set_weight(pyrosetta.rosetta.core.scoring.rama, 1.0)

        # # omega: Omega dihedral in the backbone. A Harmonic constraint on planarity with standard deviation of ~6 deg.  Supports alpha-amino acids, beta-amino acids, and oligoureas.  In the case of oligoureas, both amide bonds (called "mu" and "omega" in Rosetta) are constarined to planarity.
        # scorefxn.set_weight(pyrosetta.rosetta.core.scoring.omega, 1.0)

        energy = scorefxn(pose)
        all_energy.append(
            {
                "file_name": file_name,
                "energy": energy,
            }
        )

        # print("All energy terms and weights:")
        # for score_type in ScoreType.__members__.values():
        #     weight = scorefxn.get_weight(score_type)
        #     if weight != 0:  # 只打印权重大于0的项
        #         print(f"{score_type.name}: {weight}")

    """相同超参下所有PDB的能量均值"""
    mean_energy = sum([result["energy"] for result in all_energy]) / len(all_energy)

    """保存能量统计信息到json文件"""
    save_json(all_energy, os.path.join(save_dir, "all_energy.json"))
    save_json(mean_energy, os.path.join(save_dir, "mean_energy.json"))
    print("Energy statistics saved to json files!")


if __name__ == "__main__":
    result_dir = "/huyuqi/xmyu/FoldMLM/pdb_real/N128"
    save_dir = "/huyuqi/xmyu/FoldMLM/FoldMaskGIT/results/energy_real/N128"
    main(result_dir, save_dir)

    result_dir = "/huyuqi/xmyu/FoldMLM/pdb_real/T493"
    save_dir = "/huyuqi/xmyu/FoldMLM/FoldMaskGIT/results/energy_real/T493"
    main(result_dir, save_dir)
