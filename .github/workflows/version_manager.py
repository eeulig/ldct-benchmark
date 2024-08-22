import os
import re
import sys

# A simple script to verify/bump the version number in some files
# First argument should be patch | minor | major
# Second argument can be a semantic version number which is used to verify correctness of bump

##### CHANGE HERE
# CONFIG. Must have files to change as keys and pattern as value We need to have groups prefix->major->minor->patch->suffix
TO_CHANGE = {"pyproject.toml": '(version = ")(\d+)\.(\d+)\.(\d+)(")'}

# Whether to veify the bump, i.e. if release (provided as second argument) then check that new version is same as release version
VERIFY = True
##### STOP CHANGE


class Version:
    def __init__(self, major=None, minor=None, patch=None):
        assert major is not None, "Must provide major version!"
        assert minor is not None, "Must provide minor version!"
        assert patch is not None, "Must provide patch version!"
        # Init current version
        self.major = major
        self.minor = minor
        self.patch = patch

        # Init new version
        self.new_major, self.new_minor, self.new_patch = major, minor, patch

    def current_version(self):
        return f"{self.major}.{self.minor}.{self.patch}"

    def new_version(self):
        return f"{self.new_major}.{self.new_minor}.{self.new_patch}"

    def bump_major(self):
        self.new_major += 1
        self.new_minor = 0
        self.new_patch = 0

    def bump_minor(self):
        self.new_minor += 1
        self.new_patch = 0

    def bump_patch(self):
        self.new_patch += 1

    def bump(self, bump_type):
        if bump_type == "major":
            self.bump_major()
        elif bump_type == "minor":
            self.bump_minor()
        elif bump_type == "patch":
            self.bump_patch()
        elif bump_type == "none":
            print(
                f"[BUMPVERSION] No version bumped. Did you just want to verify an existing version? Then, nevermind!"
            )
            pass
        else:
            raise ValueError(
                f"[BUMPVERSION] Can't bump {bump_type}! Can only bump major, minor, or patch"
            )

    def verify(self, version: str):
        """Provided version must be in format major.minor.patch"""
        found = re.search("(\d+)\.(\d+)\.(\d+)", version)
        if found is None or len(found.groups()) != 3:
            raise ValueError(
                f"[BUMPVERSION] Can't get version number from provided version to verify {version}. Must be in format major.minor.patch"
            )
        major, minor, patch = int(found[1]), int(found[2]), int(found[3])

        if self.new_version() != f"{major}.{minor}.{patch}":
            raise ValueError(
                f"[BUMPVERSION] Version {self.new_version()} doesn't match desired version {major}.{minor}.{patch}!"
            )


all_updates = set()
for file, pattern in TO_CHANGE.items():
    with open(file, "r") as f:
        text = f.read()
    found = re.search(pattern, text)
    version = Version(major=int(found[2]), minor=int(found[3]), patch=int(found[4]))
    version.bump(sys.argv[1])

    if VERIFY:
        # verify that all version numbers are updated identically
        all_updates.add(version.new_version())
        if len(all_updates) > 1:
            raise ValueError(
                f"[BUMPVERSION] Would udpate to different versions in differnt files {all_updates}"
            )
        # verify that new version number matches desired one (if provided)
        if len(sys.argv) == 3:
            version.verify(sys.argv[2])

    new_file = re.sub(pattern, rf"\g<1>{version.new_version()}\g<5>", text)
    with open(file, "w") as f:
        f.write(new_file)

    if version.current_version() != version.new_version():
        print(
            f"[BUMPVERSION] Bumped version {version.current_version()} -> {version.new_version()} in {file}!"
        )

if len(all_updates) == 0 and sys.argv[1] != "none":
    raise ValueError(f"[BUMPVERSION] Did not change any version!")

# Create commit message for this version bump
if sys.argv[1] != "none":
    env_file = os.getenv("GITHUB_ENV")
    with open(env_file, "a") as myfile:
        myfile.write(
            f"BUMPVERSION_COMMIT=Bump version {version.current_version()} -> {version.new_version()}"
        )
