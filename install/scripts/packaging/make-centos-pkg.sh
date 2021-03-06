#!/bin/bash
# Intended to be executed inside the built singularity container

set -e

. ./common/init.sh

HIPSYCL_PKG_BUILD_BASE=${HIPSYCL_PKG_BUILD_BASE:-ON}
HIPSYCL_PKG_BUILD_HIPSYCL=${HIPSYCL_PKG_BUILD_HIPSYCL:-ON}
HIPSYCL_PKG_BUILD_ROCM=${HIPSYCL_PKG_BUILD_ROCM:-ON}
HIPSYCL_PKG_BUILD_CUDA=${HIPSYCL_PKG_BUILD_CUDA:-OFF}

RPM_ROOT=${BUILD_DIR}/rpm
mkdir -p ${RPM_ROOT}/{SOURCES,BUILD,RPMS,SPECS,SRPMS,tmp}


cat << EOF > ${RPM_ROOT}/SPECS/hipSYCL.spec
Summary: Implementation of Khronos SYCL for CPUs, AMD GPUs and NVIDIA GPUs
Name: hipSYCL${HIPSYCL_PKG_NAME_SUFFIX}
Version: ${HIPSYCL_VERSION}
Release: ${HIPSYCL_BUILD}
License: BSD
Packager: Aksel Alpay
Group: Development/Tools
BuildRequires: coreutils
BuildRoot: ${RPM_ROOT}/tmp/hipSYCL-${HIPSYCL_VERSION_STRING}
Requires: python3, hipSYCL-base${HIPSYCL_PKG_NAME_SUFFIX}, boost169-devel
AutoReq: no

%description
%{summary}

%install
cp -R ${HIPSYCL_DIR}/* %{buildroot}

%files
/opt/hipSYCL/bin
/opt/hipSYCL/lib
/opt/hipSYCL/include
/opt/hipSYCL/etc

EOF

cat << EOF > ${RPM_ROOT}/SPECS/hipSYCL-base.spec
Summary: base LLVM compiler stack for hipSYCL
Name: hipSYCL-base${HIPSYCL_PKG_NAME_SUFFIX}
Version: ${HIPSYCL_VERSION}
Release: ${HIPSYCL_BUILD}
License: LLVM
Packager: Aksel Alpay
Group: Development/Tools
BuildRequires: coreutils
BuildRoot: ${RPM_ROOT}/tmp/hipSYCL-base-${HIPSYCL_VERSION_STRING}
Requires: devtoolset-7

%description
%{summary}

%install
cp -R ${COMMON_DIR}/* %{buildroot}

%files
/opt/hipSYCL/llvm

EOF

cat << EOF > ${RPM_ROOT}/SPECS/hipSYCL-rocm.spec
Summary: ROCm stack for hipSYCL
Name: hipSYCL-rocm${HIPSYCL_PKG_NAME_SUFFIX}
Version: ${HIPSYCL_VERSION}
Release: ${HIPSYCL_BUILD}
License: LLVM
Packager: Aksel Alpay
Group: Development/Tools
BuildRequires: coreutils
BuildRoot: ${RPM_ROOT}/tmp/hipSYCL-rocm-${HIPSYCL_VERSION_STRING}
Requires: hipSYCL${HIPSYCL_PKG_NAME_SUFFIX}, numactl-devel, numactl-libs, pciutils-devel, pciutils-libs, perl, elfutils-libelf-devel

%description
%{summary}

%install
cp -R ${ROCM_DIR}/* %{buildroot}
  
%files
/opt/hipSYCL/rocm

EOF

cat << EOF > ${RPM_ROOT}/SPECS/hipSYCL-cuda.spec
Summary: CUDA stack for hipSYCL
Name: hipSYCL-cuda
Version: ${HIPSYCL_VERSION}
Release: ${HIPSYCL_BUILD}
License: NVIDIA CUDA EULA
Packager: Aksel Alpay
Group: Development/Tools
BuildRequires: coreutils
BuildRoot: ${RPM_ROOT}/tmp/hipSYCL-cuda-${HIPSYCL_VERSION_STRING}
AutoReq: no

%description
%{summary}

%install
cp -R ${CUDA_DIR}/* %{buildroot}


%files
/opt/hipSYCL/cuda

EOF


cd ${RPM_ROOT}/SPECS

if [ "$HIPSYCL_PKG_BUILD_HIPSYCL" = "ON"  ]; then
rpmbuild -bb hipSYCL.spec
fi

if [ "$HIPSYCL_PKG_BUILD_BASE" = "ON"  ]; then
rpmbuild -bb hipSYCL-base.spec
fi

if [ "$HIPSYCL_PKG_BUILD_ROCM" = "ON"  ]; then
rpmbuild -bb hipSYCL-rocm.spec
fi

if [ "$HIPSYCL_PKG_BUILD_CUDA" = "ON"  ]; then
rpmbuild -D '%_python_bytecompile_errors_terminate_build 0' -bb hipSYCL-cuda.spec
fi
