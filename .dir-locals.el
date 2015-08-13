;;; .dir-locals.el --- libML project local variables
;;; Commentary:
;;; Code:
((nil
  .
  ((projectile-project-compilation-cmd
    . "make -j 8 -C \"~/VersionControl/Modules/libML/build\"")
   (projectile-project-run-cmd
    . "make test -C \"~/VersionControl/Modules/libML/build\"")))
 )
;;; .dir-locals.el ends here
