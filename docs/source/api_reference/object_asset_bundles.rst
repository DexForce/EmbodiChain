Object asset archive bundles
============================

These dataset classes resolve object archives through the configured asset
hosting prefix and the standard data cache. Their class names are the first
path segment accepted by ``get_data_path`` and the names shown by
``embodichain data list --category obj``. The optional ``data_root`` constructor
argument selects a different cache root.

``PourWaterAssets`` contains the adapted bottle, cup and tabletop at the archive
root, together with source files, attribution, licenses and a SHA-256 manifest.
It uses the configured download prefix first and falls back to the official
Hugging Face repository when the prefix differs. Archive integrity is checked
against its registered MD5 by the shared dataset loader.

.. currentmodule:: embodichain.data.assets.obj_assets

.. autosummary::

   ShopTableSimple
   CircleTableSimple
   PlasticBin
   Chair
   ContainerMetal
   SimpleBoxDrawer
   AdrianoTable
   CoffeeCup
   SlidingBoxDrawer
   AluminumTable
   ToyDuck
   PaperCup
   ChainRainSec
   TableWare
   ScannedBottle
   SugarBox
   SodaCan
   MicrowaveOven
   Microwave
   PlasticTray
   WaterBasin
   Drawer
   Cow
   BakeTextureObj
   DrawerUSD
   PourWaterAssets

.. automodule:: embodichain.data.assets.obj_assets
   :members:
   :show-inheritance:
