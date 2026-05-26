# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: Add more tests for general inference inputs
import pickle
from pathlib import Path

import pytest
from biotite.structure.io import pdbx
from rdkit import Chem

from openfold3.core.data.io.structure.cif import write_structure
from openfold3.core.data.pipelines.featurization.conformer import (
    featurize_reference_conformers_of3,
)
from openfold3.core.data.primitives.structure.metadata import get_cif_block
from openfold3.core.data.primitives.structure.query import (
    processed_reference_molecule_from_mol,
    structure_with_ref_mols_from_query,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    Query,
)
from openfold3.tests.custom_assert_utils import (
    assert_atomarray_equal,
    assert_ref_mols_equal,
)

reference_data_path = Path(__file__).parent / "test_data" / "structure_from_query"

# A standard peptide query
standard_peptide_query = Query.model_validate(
    {
        "query_name": "std_peptide",
        "chains": [
            {
                "molecule_type": "protein",
                "chain_ids": "A",
                "sequence": "MACHINELEARNING",
            }
        ],
    }
)

# A peptide query with non-canonical residues methionine sulfoxide (MHO) and
# selenocysteine (SEC)
non_canonical_peptide_query = Query.model_validate(
    {
        "query_name": "non_std_peptide",
        "chains": [
            {
                "molecule_type": "protein",
                "chain_ids": "A",
                "sequence": "MACHINELEARNING",
                "non_canonical_residues": {
                    "1": "MHO",
                    "3": "SEC",
                },
            }
        ],
    }
)


@pytest.mark.parametrize(
    "query, ground_truth_file",
    [
        (
            standard_peptide_query,
            reference_data_path / "structure-w-ref-mols_std-peptide.pkl",
        ),
        (
            non_canonical_peptide_query,
            reference_data_path / "structure-w-ref-mols_non-std-peptide.pkl",
        ),
    ],
    ids=[
        "standard_peptide",
        "non_canonical_peptide",
    ],
)
def test_structure_from_query(query: Query, ground_truth_file: Path):
    """Tests that the generated structure and reference molecules matches gt."""
    structure_with_ref_mols = structure_with_ref_mols_from_query(query)

    # Get reference file
    structure_with_ref_mols_gt = pickle.loads(ground_truth_file.read_bytes())

    # Check that atom arrays match (for some reason the GT generation script generated a
    # different order of annotations but that's fine)
    assert_atomarray_equal(
        structure_with_ref_mols.atom_array,
        structure_with_ref_mols_gt.atom_array,
        strict_annot_order=False,
    )

    # Check that reference molecules match
    for ref_mol, ref_mol_gt in zip(
        structure_with_ref_mols.processed_reference_mols,
        structure_with_ref_mols_gt.processed_reference_mols,
        strict=False,
    ):
        assert_ref_mols_equal(ref_mol, ref_mol_gt)


def test_smiles_with_explicit_hydrogen():
    """Tests that SMILES with explicit hydrogens can be processed.

    Regression test for a bug where explicit hydrogens in the input molecule
    caused a length mismatch between the atom mask and the molecule after
    conformer generation (which removes hydrogens).
    """
    # SMILES with explicit hydrogen - this triggered the bug
    smiles_with_explicit_h = "[H]/C=C\\Cl"
    mol = Chem.MolFromSmiles(smiles_with_explicit_h)

    # Should not raise an error
    ref_mol = processed_reference_molecule_from_mol(mol)

    # Verify mask length matches mol atom count
    assert ref_mol.mol.GetNumAtoms() == len(ref_mol.in_crop_mask)

    # Featurization should also succeed
    features = featurize_reference_conformers_of3(
        [ref_mol],
        add_ref_space_uid_to_perm=False,
    )
    assert "ref_pos" in features


def test_smiles_ligand_cif_auth_seq_id_is_numeric(tmp_path):
    """Regression test for SMILES ligands being written with missing auth seq IDs."""
    query = Query.model_validate(
        {
            "query_name": "protein_smiles_ligand",
            "chains": [
                {
                    "molecule_type": "protein",
                    "sequence": "ACDEFGHIKLMNPQRSTVWY",
                    "chain_ids": "A",
                },
                {
                    "molecule_type": "ligand",
                    "smiles": "NCCc1cc(O)c(O)cc1",
                    "chain_ids": "X",
                },
            ],
        }
    )
    atom_array = structure_with_ref_mols_from_query(query).atom_array

    cif_path = tmp_path / "protein_smiles_ligand.cif"
    write_structure(atom_array, cif_path)

    cif_block = get_cif_block(pdbx.CIFFile.read(cif_path))
    atom_site = cif_block["atom_site"]
    ligand_mask = atom_site["label_asym_id"].as_array() == "X"

    assert ligand_mask.any()
    assert set(atom_site["label_seq_id"].as_array()[ligand_mask]) == {"."}
    assert set(atom_site["auth_seq_id"].as_array()[ligand_mask]) == {"1"}
