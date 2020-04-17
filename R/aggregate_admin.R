library(raster)
library(sf)
library(DBI)
library(RPostgreSQL)
library(glue)

# Simple way -------------------------------------------------------------------

worlpop <- raster("../Layers/pop/africa2010ppp.tif")
admin <- st_read("../Layers/Africa_SHP/Africa.shp")
zonal_sum <- extract(worlpop, admin[1,], fun = "sum", df = T)


# Postgres way -----------------------------------------------------------------
conn_pg <- RPostgres::dbConnect(RPostgres::Postgres(), 
                                dbname = "databasename") # connection to db

pop_file <- "../Layers/pop/africa2010ppp.tif"
# Write pop to database
r2psql_cmd <-glue::glue("raster2pgsql -s EPSGS:4326 -I -t auto -d {pop_file}  pop | 
                     psql -w -h localhost -d databasename")
system(r2psql_cmd)
dbSendQuery(conn_pg, "CREATE TABLE pop_cntrds AS 
    SELECT rid, x, y, geom
    FROM (
      SELECT rid, dp.* 
      FROM pop, LATERAL ST_PixelAsCentroids(rast, 1) AS dp
    ) foo;")
dbSendQuery(conn_pg, "CREATE INDEX pop_cntrds_gidx ON pop_cntrds USING GIST(geom);")
# Write admin to database
rpostgis::pgInsert(conn_pg, "admin", as_Spatial(admin), overwrite = T)
dbSendStatement(conn_pg, "UPDATE admin SET geom = ST_SetSRID(geom, 4326);")
dbSendQuery(conn_pg, "CREATE INDEX admin_gidx ON pop_cntrds USING GIST(geom);")

# Compute overlaps
# Assmuming a unique column "id" in the table "admin"
DBI::dbSendStatement(conn_pg,
                     'CREATE TABLE admin_sums AS (
                  SELECT a."ID", SUM(ST_Value(b.rast, c.geom))
                  FROM admin a, pop_cntrds c, pop b 
                  WHERE ST_Intersects(a.geom, c.geom) AND ST_Intersects(b.rast, c.geom)
                  GROUP BY a."ID" 
                );')



