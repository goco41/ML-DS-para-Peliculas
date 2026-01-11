#%%
import re
import json
import requests
import pandas as pd
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm # Importamos tqdm

def scrape_palomitacas_profile_movies(profile_movies_url: str,
                                      save_prefix: str | None = None,
                                      timeout: int = 60) -> pd.DataFrame:
    """
    Scrapea la biblioteca de películas de un usuario de Palomitacas.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120 Safari/537.36"
        ),
        "Referer": profile_movies_url,
    }

    # Descargar HTML del perfil y sacar idu
    r = requests.get(profile_movies_url, headers=headers, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    sel = soup.select_one("select.filter-movies[data-idu]")
    if not sel or not sel.get("data-idu"):
        raise RuntimeError("No se encuentra data-idu. ¿Perfil privado o URL incorrecta?")
    idu = sel["data-idu"]

    btn = soup.select_one(".show-more-items")
    if not btn:
        itemsrc = "movies"
        step = 50
        sortby = "ua"
    else:
        itemsrc = btn.get("data-itemsrc") or "movies"
        step = int(btn.get("data-offset") or 50)
        sortby = btn.get("data-sortby") or "ua"

    base = f"{urlparse(profile_movies_url).scheme}://{urlparse(profile_movies_url).netloc}"

    def tmdb_from_url(u: str):
        m = re.search(r"/pelicula/ficha/p(\d+)", u or "")
        return int(m.group(1)) if m else None

    # Parsear las primeras cards
    def parse_catalog_cards_from_soup(s: BeautifulSoup):
        rows = []
        for c in s.select("div.box-peli-small-biblioteca"):
            title = (c.get("data-hint") or "").strip() or None
            a = c.select_one("a[href]")
            url = urljoin(base + "/", a["href"]) if a and a.get("href") else (c.get("data-url") or None)
            if not url:
                continue
            rows.append({"title": title, "url": url, "tmdb_id": tmdb_from_url(url)})
        return rows

    catalog_rows = parse_catalog_cards_from_soup(soup)

    # Paginación
    offset = step
    while btn: 
        payload = {"offset": str(offset), "tipo": itemsrc, "idu": str(idu), "sortby": sortby}
        rr = requests.post(base + "/perfil/ajax_show_more", headers=headers, data=payload, timeout=timeout)
        rr.raise_for_status()

        data = json.loads(rr.text)
        fragment = data.get("respuesta", "")
        if not isinstance(fragment, str) or not fragment.strip():
            break

        part_soup = BeautifulSoup(fragment, "html.parser")
        part = parse_catalog_cards_from_soup(part_soup)

        if not part:
            break

        catalog_rows.extend(part)

        if len(part) < step:
            break

        offset += step
        if offset > 25000: # límite de seguridad
            break

    df_catalog = (
        pd.DataFrame(catalog_rows)
        .dropna(subset=["url"])
        .drop_duplicates(subset=["url"])
        .reset_index(drop=True)
    )

    # Checkins (notas y fechas)
    rr = requests.post(
        base + "/perfil/ajax_filtro_orden_peliculas",
        headers=headers,
        data={"idu": str(idu), "filtro": "checkin"},
        timeout=timeout
    )
    rr.raise_for_status()
    data = json.loads(rr.text)
    soup_checkin = BeautifulSoup(data.get("respuesta", "") or "", "html.parser")

    def to_float(x):
        if x is None: return None
        x = str(x).strip().replace(",", ".")
        if x == "": return None
        try: return float(x)
        except: return None

    rows_checkin = []
    for c in soup_checkin.select("div.box-peli-small-biblioteca"):
        url = c.get("data-url") or None
        if not url: continue
        
        user_rating = to_float(c.get("data-irate"))
        if user_rating is None:
            vu = c.select_one(".valoracion-usuario")
            if vu:
                user_rating = to_float(vu.get_text(" ", strip=True).split()[0])
        
        rows_checkin.append({
            "url": url,
            "tmdb_id": to_float(c.get("data-idp")),
            "user_rating": user_rating,
            "site_rating": to_float(c.get("data-mrate")),
            "added_label": c.get("data-added") or None,
            "checkin_date": c.get("data-checkin") or None,
        })

    df_checkin = pd.DataFrame(rows_checkin)
    if not df_checkin.empty:
        df_checkin = df_checkin.dropna(subset=["url"]).drop_duplicates(subset=["url"])
        df_checkin["tmdb_id"] = pd.to_numeric(df_checkin["tmdb_id"], errors="coerce").astype("Int64")
    
    # Merge
    if df_checkin.empty:
        df_final = df_catalog
        for col in ["user_rating", "site_rating", "added_label", "checkin_date"]:
            df_final[col] = None
    else:
        df_final = df_catalog.merge(
            df_checkin[["url", "user_rating", "site_rating", "added_label", "checkin_date", "tmdb_id"]],
            on="url",
            how="left",
            suffixes=("", "_chk")
        )
        df_final["tmdb_id"] = df_final["tmdb_id"].fillna(df_final["tmdb_id_chk"])
        df_final = df_final.drop(columns=["tmdb_id_chk"])

    # Guardar
    if save_prefix:
        path_csv = f"{save_prefix}.csv"
        df_final.to_csv(path_csv, index=False, encoding="utf-8")

    return df_final


# BLOQUE DE EJECUCIÓN CON CSV

# Rutas de configuración
INPUT_CSV = os.path.join("scraper", "links.csv")
OUTPUT_FOLDER = os.path.join("data", "valoraciones")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Contadores de éxito y fallo
successful_scrapes = 0
failed_scrapes = 0

# Leer el CSV
try:
    print(f"Leyendo archivo de links: {INPUT_CSV}")
    df_links = pd.read_csv(INPUT_CSV)
    
    if "Link" not in df_links.columns:
        raise ValueError("El CSV no tiene una columna llamada 'Link'.")
        
    links_list = df_links["Link"].dropna().astype(str).tolist()
    print(f"Se han encontrado {len(links_list)} links para procesar.\n")

except Exception as e:
    print(f"Error crítico leyendo el CSV: {e}")
    links_list = []

total_links = len(links_list)

# Procesar cada link con tqdm
for base_url in tqdm(links_list, desc="Procesando usuarios"):
    base_url = base_url.strip()
    if not base_url:
        failed_scrapes += 1
        continue
    
    base_url_clean = base_url.rstrip("/")
    
    username = base_url_clean.split("/")[-1]

    movies_url = f"{base_url_clean}/peliculas"
    
    save_path = os.path.join(OUTPUT_FOLDER, f"{username}_valoraciones")
    
    try:
        scrape_palomitacas_profile_movies(
            profile_movies_url=movies_url,
            save_prefix=save_path
        )
        successful_scrapes += 1
        
    except Exception as e:
        failed_scrapes += 1
        tqdm.write(f"   [ERROR] Falló el usuario {username}. URL: {movies_url}")
        tqdm.write(f"   Motivo: {e}")
        continue

print("\n--- Proceso finalizado ---")

# Reporte final
print(f"Total de enlaces intentados: {total_links}")
print(f"Respuestas exitosas: {successful_scrapes}")
print(f"Respuestas fallidas: {failed_scrapes}")

if total_links > 0:
    success_percentage = (successful_scrapes / total_links) * 100
    print(f"Porcentaje de éxito: {success_percentage:.2f}%")
else:
    print("No se procesaron enlaces.")
# %%

