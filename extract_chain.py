from Bio import PDB
import os

# 创建一个解析器
parser = PDB.PDBParser()

# 读取PDB文件
structure = parser.get_structure('protein', 'your_pdb_file.pdb')

# 创建保存子链的目录
output_dir = "chains_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历结构中的所有子链
for model in structure:
    for chain in model:
        # 获取子链ID
        chain_id = chain.get_id()
        # 计算子链中氨基酸的数量
        residue_count = len([residue for residue in chain if PDB.is_aa(residue)])

        # 根据氨基酸数量分类
        output_file = os.path.join(output_dir, f"chain_{chain_id}_residues_{residue_count}.pdb")

        # 保存子链到新文件
        with open(output_file, "w") as f:
            # 用PDBIO将子链写入新文件
            io = PDB.PDBIO()
            io.set_structure(chain)
            io.save(f)

        print(f"Saved chain {chain_id} with {residue_count} residues to {output_file}")
