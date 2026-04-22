"""CLAWHUB Registry tools for skill management.

This module provides tools for searching, installing, and dynamically loading
skills from the CLAWHUB_REGISTRY.
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import httpx
from langchain_core.tools import tool
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logging import logger


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
async def _make_registry_request(endpoint: str, params: Optional[dict] = None) -> dict:
    """Make an HTTP request to the CLAWHUB_REGISTRY with retry logic.

    Args:
        endpoint: The API endpoint to call.
        params: Optional query parameters.

    Returns:
        dict: The JSON response from the registry.

    Raises:
        Exception: If the request fails after all retries.
    """
    url = f"{settings.CLAWHUB_REGISTRY_URL}{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            "registry_http_error",
            url=url,
            status_code=e.response.status_code,
            error=str(e),
        )
        raise Exception(f"Registry request failed with status {e.response.status_code}: {str(e)}")
    except httpx.RequestError as e:
        logger.error("registry_request_error", url=url, error=str(e))
        raise Exception(f"Registry request failed: {str(e)}")
    except Exception as e:
        logger.error("registry_request_unexpected_error", url=url, error=str(e))
        raise Exception(f"Unexpected error calling registry: {str(e)}")


@tool
async def search_clawhub_skills(query: str, page: int = 0, limit: int = 10) -> str:
    """Search for skills in the CLAWHUB_REGISTRY.

    Use this tool to find available skills that can be installed and loaded.
    Returns a list of matching skills with their metadata.

    Args:
        query: Search query to find skills (e.g., "weather", "calculator", "data_analysis").
        page: Page number for pagination (default: 0).
        limit: Maximum number of results to return per page (default: 10).

    Returns:
        str: JSON string containing search results with skill metadata.
    """
    try:
        logger.info("searching_clawhub_skills", query=query, page=page, limit=limit)
        
        params = {"q": query, "page": page, "limit": limit}
        result = await _make_registry_request("/api/v1/search", params)
        
        if not result or "results" not in result:
            logger.warning("no_skills_found", query=query)
            return json.dumps({"results": [], "total": 0, "page": page, "limit": limit, "message": "No skills found matching the query."})
        
        results = result["results"]
        total = result.get("total", 0)
        logger.info("clawhub_skills_search_completed", query=query, result_count=len(results), total=total)
        
        return json.dumps(
            {
                "results": results,
                "total": total,
                "page": page,
                "limit": limit,
                "message": f"Found {len(results)} skills matching '{query}' (total: {total})"
            },
            indent=2
        )
    except Exception as e:
        logger.exception("search_clawhub_skills_failed", query=query, error=str(e))
        return json.dumps({"error": f"Failed to search skills: {str(e)}"})


@tool
async def install_clawhub_skill(slug: str, version: Optional[str] = None) -> str:
    """Install a skill from CLAWHUB_REGISTRY to the local skills directory.

    Use this tool to download and install a skill package. The skill will be
    installed in the configured CLAWHUB_SKILLS_DIR.

    Args:
        slug: The slug identifier of the skill to install (e.g., "namespace--skill_name").
        version: Optional version to install (defaults to latest if not specified).

    Returns:
        str: Installation result message.
    """
    try:
        logger.info("installing_clawhub_skill", slug=slug, version=version)
        
        # Ensure skills directory exists
        skills_dir = settings.CLAWHUB_SKILLS_DIR
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine download strategy based on whether version is provided
        if version:
            # If version is provided, directly download using version
            resolved_version = version
        else:
            # If version is empty, call /api/v1/resolve to get latest version
            params = {"slug": slug}
            resolve_result = await _make_registry_request("/api/v1/resolve", params)
            
            if not resolve_result:
                logger.error("skill_resolve_failed", slug=slug)
                return json.dumps({"error": f"Skill '{slug}' not found in registry or resolve failed."})
            
            # Extract version from resolve response
            if "latestVersion" in resolve_result and "version" in resolve_result["latestVersion"]:
                resolved_version = resolve_result["latestVersion"]["version"]
            else:
                logger.error("skill_resolve_no_version", slug=slug, resolve_result=resolve_result)
                return json.dumps({"error": f"Skill '{slug}' resolve response missing version information."})
        
        # Parse slug to extract namespace and skill name (format: namespace--skill_name)
        if "--" in slug:
            namespace, skill_name = slug.split("--", 1)
        else:
            namespace = "global"
            skill_name = slug
        
        # Construct download URL using /api/v1/skills/{namespace}/{skill_name}/versions/{version}/download
        download_url = f"{settings.CLAWHUB_REGISTRY_URL}/api/v1/skills/{namespace}/{skill_name}/versions/{resolved_version}/download"
        logger.info("downloading_skill", slug=slug, namespace=namespace, skill_name=skill_name, version=resolved_version, download_url=download_url)
        
        # Download the skill package
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(download_url)
            response.raise_for_status()
            
            # Detect file type from Content-Type header
            content_type = response.headers.get("Content-Type", "")
            logger.info("download_response_content_type", content_type=content_type)
            
            # Determine file extension based on content type
            if "zip" in content_type.lower():
                file_ext = ".zip"
            elif "tar" in content_type.lower() or "gzip" in content_type.lower():
                file_ext = ".tar.gz"
            else:
                # Default to .zip if content type is unclear
                file_ext = ".zip"
                logger.warning("unknown_content_type_using_default", content_type=content_type, default_ext=".zip")
            
            # Save the downloaded file
            package_filename = f"{slug.replace('/', '-')}_{resolved_version}{file_ext}"
            package_path = skills_dir / package_filename
            
            with open(package_path, "wb") as f:
                f.write(response.content)
            
            logger.info("skill_package_downloaded", slug=slug, version=resolved_version, path=str(package_path), file_ext=file_ext)
        
        # Extract the package
        skill_dir_name = slug.replace("--", "_")
        extract_dir = skills_dir / skill_dir_name
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        
        shutil.unpack_archive(package_path, extract_dir)
        logger.info("skill_package_extracted", slug=slug, extract_dir=str(extract_dir))
        
        # Clean up the downloaded package
        package_path.unlink()
        
        logger.info("clawhub_skill_installed", slug=slug, version=resolved_version)
        
        return json.dumps({
            "success": True,
            "slug": slug,
            "version": resolved_version,
            "install_path": str(extract_dir),
            "message": f"Successfully installed '{slug}' version {resolved_version}"
        })
    except Exception as e:
        logger.exception("install_clawhub_skill_failed", slug=slug, error=str(e))
        return json.dumps({"error": f"Failed to install skill '{slug}': {str(e)}"})


@tool
async def load_clawhub_skill(slug: str) -> str:
    """Read the SKILL.md content from an installed skill.

    Use this tool to read the skill documentation from the local skills directory.
    The skill must be installed first using install_clawhub_skill.

    Args:
        slug: The skill name to load (e.g., "skill_name")

    Returns:
        str: SKILL.md content or error message.
    """
    try:
        logger.info("loading_clawhub_skill", slug=slug)
        
        skills_dir = settings.CLAWHUB_SKILLS_DIR
        skill_dir_name = slug.replace("--", "_")
        skill_dir = skills_dir / skill_dir_name
        
        if not skill_dir.exists():
            logger.error("skill_directory_not_found", slug=slug, skills_dir=str(skills_dir))
            return json.dumps({"error": f"Skill directory not found: {skill_dir}. Please install the skill first."})
        
        skill_md_path = skill_dir / "SKILL.md"
        
        if not skill_md_path.exists():
            logger.error("skill_md_not_found", slug=slug, skill_dir=str(skill_dir))
            return json.dumps({"error": f"SKILL.md not found in {skill_dir}"})
        
        with open(skill_md_path, "r", encoding="utf-8") as f:
            skill_md_content = f.read()
        
        logger.info("skill_md_loaded", slug=slug, path=str(skill_md_path))
        
        return json.dumps({
            "success": True,
            "slug": slug,
            "content": skill_md_content,
            "message": f"Successfully loaded SKILL.md for '{slug}'"
        })
    except Exception as e:
        logger.exception("load_clawhub_skill_failed", slug=slug, error=str(e))
        return json.dumps({"error": f"Failed to load skill '{slug}': {str(e)}"})


@tool
async def list_installed_clawhub_skills() -> str:
    """List all installed skills from the local skills directory.

    Use this tool to see which skills are currently installed and available
    for loading.

    Returns:
        str: JSON string containing list of installed skills.
    """
    try:
        logger.info("listing_installed_clawhub_skills")
        
        skills_dir = settings.CLAWHUB_SKILLS_DIR
        
        if not skills_dir.exists():
            logger.warning("skills_directory_not_exists", skills_dir=str(skills_dir))
            return json.dumps({"skills": [], "message": "Skills directory does not exist. No skills installed."})
        
        installed_skills = []
        for skill_path in skills_dir.iterdir():
            if skill_path.is_dir() and not skill_path.name.startswith("_"):
                skill_info = {
                    "name": skill_path.name,
                    "path": str(skill_path),
                    "exists": True
                }
                
                # Try to read SKILL.md frontmatter
                skill_md_path = skill_path / "SKILL.md"
                if skill_md_path.exists():
                    try:
                        with open(skill_md_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            # Extract frontmatter between --- ---
                            if content.startswith("---"):
                                end_marker = content.find("---", 3)
                                if end_marker != -1:
                                    frontmatter = content[3:end_marker].strip()
                                    skill_info["summary"] = frontmatter
                    except Exception as e:
                        logger.warning("failed_to_read_skill_md", skill_name=skill_path.name, error=str(e))
                
                installed_skills.append(skill_info)
        
        logger.info("installed_skills_listed", count=len(installed_skills))
        
        return json.dumps({
            "skills": installed_skills,
            "count": len(installed_skills),
            "message": f"Found {len(installed_skills)} installed skills"
        }, indent=2)
    except Exception as e:
        logger.exception("list_installed_clawhub_skills_failed", error=str(e))
        return json.dumps({"error": f"Failed to list installed skills: {str(e)}"})
