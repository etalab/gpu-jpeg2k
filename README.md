# gpu-jpeg2k

See http://apps.man.poznan.pl/trac/jpeg2k/wiki for a description of the project.

When I tried to use the project with CUDA 8.0 I stumbled upon a compilation issue. Since the original maintainer has not worked on the project during the past 4 years, I decided to publish a fix by forking the original project with GitHub.

## Installation

Compilation succeeds with the following version of `gcc` : 4.9.4, 5.3.0, 5.4.1. Compilation fails with `gcc` 6.3.

Follow the installation instructions in the original wiki page. But instead of cloning the svn repository, clone this repository.
