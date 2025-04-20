target("lc-hlsl-codegen")
_config_project({
    project_kind = "static",
    batch_size = 2
})
add_deps("lc-vstl")
add_files("**.cpp")
set_pcxxheader("pch.h")
add_headerfiles("*.h")
if get_config("lc_xrepo_dir") then
    add_packages("zlib", {
        public = false,
        inherit = false
    })
else
    add_deps("zlib")
end
target_end()