# MySQL Configuration
Ensure that the MySQL global variables `innod_db_buffer_pool_size` and `innodb_buffer_pool_instances` are set adequately high.
On Brandon's Windows machine, `innod_db_buffer_pool_size=4G` and `innodb_buffer_pool_instances=16`

These settings can be configured via the options file. On Windows:
1. Run "services.msc" (either from start menu search box or using the "run" utility)
2. Find MySQL in the list, _e.g._ MySQL57
3. Right click on the row with MySQL in it, select "Properties"
4. Under "Path to executable" locate the "--defaults-file" input
5. Open the "--defaults-file," search for "innodb"
6. Alter necessary defaults.
7. Restart MySQL.

Alternatively, these setting can be tweaked in MySQL Workbench the "Options File" button under the "Instance" heading in the "Navigator" pane (on the left edge of the program).
