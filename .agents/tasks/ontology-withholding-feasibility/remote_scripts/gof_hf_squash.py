"""Squash the history of the GO-function results dataset to one commit, then delete
LFS objects that no current file references (old resume.pt snapshots etc.)."""
from huggingface_hub import HfApi
R = "olukotunjosh/cdcl-go-function-results"
a = HfApi()
def current():
    return {f.path: f for f in a.list_repo_tree(R, repo_type="dataset", recursive=True, expand=True)
            if getattr(f, "size", None) is not None}
before = current()
keep = {f.lfs.sha256 for f in before.values() if getattr(f, "lfs", None)}
a.super_squash_history(R, repo_type="dataset", commit_message="squash history (current files kept)")
after = current()
assert {p: (f.blob_id, f.size) for p, f in before.items()} == {p: (f.blob_id, f.size) for p, f in after.items()}, "current files changed"
orphans = [x for x in a.list_lfs_files(R, repo_type="dataset") if x.file_oid not in keep]
if orphans:
    a.permanently_delete_lfs_files(R, orphans, repo_type="dataset", rewrite_history=False)
print(f"commits {len(a.list_repo_commits(R, repo_type='dataset'))}; current files {len(after)} "
      f"({sum(f.size for f in after.values())/1e9:.1f} GB); deleted {len(orphans)} orphan LFS objects "
      f"({sum(x.size for x in orphans)/1e9:.1f} GB)")
