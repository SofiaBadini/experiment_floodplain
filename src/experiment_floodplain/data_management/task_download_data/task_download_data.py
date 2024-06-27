"""This script downloads publicly available data necessary for the replication of this paper 
(available via `4TU.ResearchData`_) and saves them to *bld/replication_data*. The datasets are the following:

  .. _4TU.ResearchData: https://data.4tu.nl/datasets/46fa1840-6522-45cd-b36c-038b259c4c95

- **BAG data**: The *BAG* folder contains data about all the (full) addresses in the Netherlands, 
  derived from the `Addresses and Buildings Key Registry`_ (*Basisregistraties Adressen en Gebouwen*, or BAG) 
  and acquired in .csv format via `Geotoko`_, in July 2022. 
  
  The .csv file provides a variety of building-related information, such as the function of each 
  (dwelling within a) building, its perimeter and size, and its year of construction. 
  The dataset codebook (in Dutch) is included in the folder.

  .. _Addresses and Buildings Key Registry: https://business.gov.nl/regulation/addresses-and-buildings-key-geo-register/
  .. _Geotoko: https://geotoko.nl/

- **ENW data**: The *ENW* folder contains shapefiles representing those areas of the 
  Southern regions of The Netherlands (Limburg and North Brabant) that were flooded in July 2021.
  These maps have been shared by the  `ENW`_ (*Expertise Netwerk Waterveiligheid*), the association 
  of Dutch flood protection specialists, via the 4TU.ResearchData repository.

  The maps are based on best available information collected via aerial photography and during 
  fieldwork at the flooded sites, and include realized inundation extent (in the sub-folders 
  *floodsGeul*, *floodsMaas*, and *floodsRoer*), areas evacuated via emergency ordinances (in the 
  sub-folder *evacuations*), and locations where incidents to water management infrastructure occurred 
  (in the sub-folder *incidents*).

  .. _ENW: https://www.enwinfo.nl/

- **Risicokaart data**: The *RISICOKAART* folder contains shapefiles of the flood maps developed for the European Floods
  Directive (ROR2) delivered at the end of 2019, obtained in October 2021 by contacting 
  `lbo@risicokaart.nl`_. The layer can be viewed by the general population via an interactive 
  app available at the `Risicokaart`_ website. 
  
  Flood maps have been developed for four different scenarios: Large probability 
  (10% risk in any given year), medium probability (1% risk in any given year), small probability 
  (0.1% risk in any given year, or 1 in 1,000 years flood), and scenario of extraordinary events 
  (0.01% risk in any given year, or or 1 in 10,000 years flood). Data on predicted flood extents 
  under each scenario, represented by polygons classified by maximum water depth, are in the folder 
  *floodmaps2019*.

  The folder *otherinfo* contains additional data on the location of primary water
  defences (which protects The Netherlands against floods from major rivers and
  the sea) and of regional water defences (which protects The Netherlands against
  floods from smaller rivers).

  .. _Risicokaart: https://www.risicokaart.nl/
  .. _lbo@risicokaart.nl: lbo@risicokaart.nl

- **Survey data**: The *SURVEY* folder contains the dataset of survey recipients without identifying
  information (e.g., no addresses), named ``survey_recipients.csv``,
  and a synthentic dataset named ``synthetic_survey_data.csv`` generated using `Synthetic Data Vault (SDV)`_
  and based on the original Qualtrics data. 
  
  .. note:: 
    
    This project automatically uses the real survey respondents' data if the corresponding 
    file is placed under *src/experiment_floodplain/data*. See also the Introduction of this 
    documentation.

  The *SURVEY* folder contains two more datasets: ``original_variables.csv``, which details which
  original variables will be cleaned (where by original I mean the "raw" data collected by 
  Qualtrics), and ``final_variables.csv``, which details which variables will appear 
  in the final dataset. 

  .. _Synthetic Data Vault (SDV): https://sdv.dev/

"""

import pytask
import requests, zipfile, io

from experiment_floodplain.config import BLD

SURVEY = BLD / "replication_data" / "SURVEY"
BAG = BLD / "replication_data" / "BAG"
ENW = BLD / "replication_data" / "ENW" / "floods-july-2021"
RISICOKAART = BLD / "replication_data" / "RISICOKAART" / "floodmaps2019"
produces = {
  "survey_recipients": SURVEY / "survey_recipients.csv",
  "synthetic_survey_data": SURVEY / "synthetic_survey_data.csv",
  "original_variables": SURVEY / "original_variables.csv",
  "final_variables": SURVEY / "final_variables.csv",
  "bag" : BAG / "bag-adressen-woning-nl.csv.zip",
  "geul": ENW / "floodsGeul" / "GeulFloodExtent.shp",
  "maas": ENW / "floodsMaas" / "MaasFloodExtent.shp",
  "roer": ENW / "floodsRoer" / "RoerFloodExtent.shp",
  "incidents": ENW / "incidents" / "incidents.shp",
  "evacuations": ENW / "evacuations" / "evacuations.shp",
  "rk10": RISICOKAART / "10depth" / "10depth.shp",
  "rk100": RISICOKAART / "100depth" / "100depth.shp",
  "rk1000": RISICOKAART / "1000depth" / "1000depth.shp",
  "rk10000": RISICOKAART / "10000depth" / "10000depth.shp",
}

@pytask.mark.produces(produces)
def task_download_data(depends_on, produces):

    zip_file_url = "https://data.4tu.nl/file/46fa1840-6522-45cd-b36c-038b259c4c95/9f442f0f-e3eb-40c0-890e-b70184024c38"
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(BLD)