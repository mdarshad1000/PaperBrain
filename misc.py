import pkg_resources

package_name = "whitenoise"

try:
    package = pkg_resources.get_distribution(package_name)
    version = package.version
    print(f"The version of {package_name} is {version}")
except pkg_resources.DistributionNotFound:
    print(f"The package {package_name} is not installed")
