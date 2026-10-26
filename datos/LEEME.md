# Datos de entrada

| Archivo | Contenido | Tamaño |
|---|---|---|
| `Clinical_and_Other_Features.xlsx` | Variables clínicas, anatomopatológicas, de tratamiento, de respuesta y de adquisición de la RM (922 pacientes) | 0,6 MB |
| `Imaging_Features.xlsx` | 529 características de imagen extraídas por los autores de la colección a partir de la DCE-MRI pretratamiento | 6,7 MB |

Ambos archivos se reproducen **sin modificaciones** (idénticos byte a byte a los publicados) desde la colección pública *Duke-Breast-Cancer-MRI* de The Cancer Imaging Archive (TCIA). SHA-256: `8ef0945c9f7513acd2d9c6e8866d98f2234dd78443110d9496c24a82dbbc7e6f` (clínico) y `371e7aec937ccf7cff01bc3c66063ce22d9699d3d3f8c6ac0dcee6728826682d` (imaging features), verificados el 25/10/2026. Al leer el Excel clínico, `pipeline_v2/run_stage2.leer_clinico` descarta la fila superior de grupos y asigna nombres únicos a 16 columnas cuyo nombre figura en esa fila o en la leyenda; los valores no se modifican. La segmentación del tumor y la extracción de las características radiómicas fueron realizadas por los autores de la colección, no en este trabajo.

## Licencia y condiciones de uso

- Licencia de ambos archivos: **Creative Commons Atribución-NoComercial 4.0 Internacional (CC BY-NC 4.0)** — <https://creativecommons.org/licenses/by-nc/4.0/>. Permite copiar y redistribuir con atribución y sin fines comerciales.
- Su uso está sujeto a la *Data Usage Policy and Restrictions* de TCIA: <https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/>. Entre otras condiciones, prohíbe intentar identificar o contactar a las participantes.
- La licencia CC BY-NC 4.0 de los datos es independiente de la licencia del código de este repositorio.

## Citas requeridas

Cita de los datos (obligatoria, con DOI):

> Saha, A., Harowicz, M. R., Grimm, L. J., Weng, J., Cain, E. H., Kim, C. E., Ghate, S. V., Walsh, R., & Mazurowski, M. A. (2021). *Dynamic contrast-enhanced magnetic resonance images of breast cancer patients with tumor locations* [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.e3sv-re93

Publicación asociada:

> Saha, A., Harowicz, M. R., Grimm, L. J., Kim, C. E., Ghate, S. V., Walsh, R., & Mazurowski, M. A. (2018). A machine learning approach to radiogenomics of breast cancer: a study of 922 subjects and 529 DCE-MRI features. *British Journal of Cancer, 119*(4), 508–516. https://doi.org/10.1038/s41416-018-0185-8

## Obtención desde la fuente oficial

Si los archivos no están presentes, se descargan desde la página de la colección (<https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/>), en la sección de datos clínicos e imaging features, y se colocan en esta carpeta con los nombres indicados. Otra ubicación puede indicarse con la variable de entorno `RADIOMICS_DATOS`.
