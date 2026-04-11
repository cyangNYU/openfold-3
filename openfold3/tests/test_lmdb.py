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

"""Tests for LMDB dict and multiprocessing safety."""

import json
import pickle
import sys

import lmdb
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from openfold3.core.data.io.dataset_cache import read_datacache
from openfold3.core.data.primitives.caches.lmdb import (
    LMDBDict,
    LMDBEnv,
    convert_datacache_to_lmdb,
)

TEST_DATASET_CONFIG = {
    "_type": "ProteinMonomerDatasetCache",
    "name": "DummySet",
    "structure_data": {
        "test0": {
            "chains": {
                "0": {
                    "alignment_representative_id": "test_id0",
                    "template_ids": [],
                    "index": 0,
                },
            },
        },
        "test1": {
            "chains": {
                "0": {
                    "alignment_representative_id": "test_id1",
                    "template_ids": [],
                    "index": 1,
                },
            },
        },
    },
    "reference_molecule_data": {
        "ALA": {
            "conformer_gen_strategy": "default",
            "fallback_conformer_pdb_id": None,
            "canonical_smiles": "C[C@H](N)C(=O)O",
            "set_fallback_to_nan": False,
        },
    },
}


def create_test_lmdb(lmdb_dir, num_items=10):
    """Create a small LMDB with ``item:0`` … ``item:N-1`` keys."""
    env = lmdb.open(str(lmdb_dir), map_size=1024 * 1024, subdir=True)
    with env.begin(write=True) as txn:
        for i in range(num_items):
            key = f"item:{i}".encode()
            value = json.dumps({"index": i}).encode("utf-8")
            txn.put(key, value)
    env.close()


class LMDBDataset(Dataset):
    """Minimal Dataset backed by LMDBEnv + LMDBDict.

    Defined at module level so spawn/forkserver workers can import it.
    """

    def __init__(self, lmdb_dir: str, num_items: int):
        self._lmdb_env = LMDBEnv(lmdb_dir)
        self._dict = LMDBDict(
            lmdb_env=self._lmdb_env,
            prefix="item",
            key_encoding="utf-8",
            value_encoding="utf-8",
        )
        self._n = num_items

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        item = self._dict[str(idx)]
        return torch.tensor(item["index"])

    def release_connections(self):
        self._lmdb_env.close()


class TestLMDBDict:
    def test_lmdb_roundtrip(self, tmp_path):
        # Save dummy json
        test_config_json = tmp_path / "test_config.json"
        with open(test_config_json, "w") as f:
            json.dump(TEST_DATASET_CONFIG, f, indent=4)

        # Create LMDB
        test_lmdb_dir = tmp_path / "test_lmdb"
        map_size = 20 * 1024
        convert_datacache_to_lmdb(test_config_json, test_lmdb_dir, map_size)

        # read lmdb
        lmdb_cache = read_datacache(test_lmdb_dir)
        # compare with json reloaded cache
        expected_cache = read_datacache(test_config_json)

        assert lmdb_cache == expected_cache


class TestLMDBEnvPickle:
    def test_raw_lmdb_env_not_pickleable(self, tmp_path):
        """Raw lmdb.Environment cannot be pickled — this is the root cause
        of spawn/forkserver failures without the LMDBEnv wrapper."""
        lmdb_dir = tmp_path / "raw"
        create_test_lmdb(lmdb_dir)
        env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, subdir=True)
        with pytest.raises(TypeError, match="cannot pickle"):
            pickle.dumps(env)
        env.close()

    def test_lmdb_env_pickle_roundtrip(self, tmp_path):
        """LMDBEnv can be pickled and reads correctly after unpickling."""
        lmdb_dir = tmp_path / "env_pkl"
        create_test_lmdb(lmdb_dir, num_items=3)

        env = LMDBEnv(str(lmdb_dir))
        _ = env.get()  # force open

        data = pickle.dumps(env)
        env.close()  # close original — LMDB forbids two open envs for same path

        env2 = pickle.loads(data)
        assert env2._env is None  # connection stripped by __getstate__
        with env2.get().begin() as txn:
            assert txn.get(b"item:0") is not None
        env2.close()

    def test_lmdb_dict_pickle_roundtrip(self, tmp_path):
        """LMDBDict survives pickle roundtrip and reads correctly."""
        lmdb_dir = tmp_path / "dict_pkl"
        num_items = 5
        create_test_lmdb(lmdb_dir, num_items=num_items)

        env = LMDBEnv(str(lmdb_dir))
        d = LMDBDict(
            lmdb_env=env,
            prefix="item",
            key_encoding="utf-8",
            value_encoding="utf-8",
        )
        original = d["0"]

        data = pickle.dumps(d)
        env.close()  # close original — LMDB forbids two open envs for same path

        d2 = pickle.loads(data)
        assert d2["0"] == original
        assert len(d2) == num_items


class TestLMDBMultiprocessingDataLoader:
    @pytest.mark.parametrize("mp_context", ["fork", "forkserver", "spawn"])
    def test_dataloader_reads_all_items(self, tmp_path, mp_context):
        """DataLoader with num_workers>0 reads all LMDB items correctly
        across fork, forkserver, and spawn multiprocessing contexts."""
        if mp_context == "fork" and sys.platform == "darwin":
            pytest.skip("fork is unsafe on macOS with Python >= 3.8")

        num_items = 20
        lmdb_dir = tmp_path / "mp_lmdb"
        create_test_lmdb(lmdb_dir, num_items=num_items)

        dataset = LMDBDataset(str(lmdb_dir), num_items=num_items)
        dataset.release_connections()  # mimic real codebase: clean state before fork

        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=2,
            multiprocessing_context=mp_context,
        )

        results = []
        for batch in loader:
            results.extend(batch.tolist())

        assert sorted(results) == list(range(num_items))
