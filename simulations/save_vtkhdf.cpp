#include <array>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <source_location>
#include <span>
#include <stdexcept>

#include <Kokkos_Core.hpp>
#include <hdf5.h>

#include "save_vtkhdf.hpp"

namespace {

void h5check(herr_t value, std::source_location const location = std::source_location::current())
{
    if (value < 0) {
        std::cerr << "file: " << location.file_name() << '(' << location.line() << ':'
                  << location.column() << ") `" << location.function_name() << "` returned "
                  << value << '\n';
        std::abort();
    }
}

class RaiiH5Hid
{
private:
    hid_t m_id;

    std::function<herr_t(hid_t)> m_close;

public:
    RaiiH5Hid(hid_t id, herr_t (*f)(hid_t)) : m_id(id), m_close(f)
    {
        if (m_id < 0 || !m_close) {
            throw std::runtime_error("Error: creating h5 id failed");
        }
    }

    RaiiH5Hid(RaiiH5Hid const&) = delete;

    RaiiH5Hid(RaiiH5Hid&&) = delete;

    ~RaiiH5Hid() noexcept
    {
        if (m_id >= 0 && m_close) {
            m_close(m_id);
        }
    }

    auto operator=(RaiiH5Hid const&) -> RaiiH5Hid& = delete;

    auto operator=(RaiiH5Hid&&) -> RaiiH5Hid& = delete;

    auto operator*() const noexcept -> hid_t
    {
        return m_id;
    }
};

void write_attribute(
        RaiiH5Hid const& object_id,
        char const* const attribute_name,
        std::string_view const attribute_value,
        H5T_cset_t const cset)
{
    RaiiH5Hid const space_id(::H5Screate(H5S_SCALAR), ::H5Sclose);

    RaiiH5Hid const type_id(::H5Tcopy(H5T_C_S1), ::H5Tclose);
    h5check(::H5Tset_size(*type_id, attribute_value.size()));
    h5check(::H5Tset_cset(*type_id, cset));

    RaiiH5Hid const attr_id(
            ::H5Acreate2(*object_id, attribute_name, *type_id, *space_id, H5P_DEFAULT, H5P_DEFAULT),
            ::H5Aclose);

    h5check(::H5Awrite(*attr_id, *type_id, attribute_value.data()));
}

void write_attribute(
        RaiiH5Hid const& object_id,
        char const* const attribute_name,
        std::span<int const> const& attribute_value)
{
    std::array<hsize_t, 1> const dims {attribute_value.size()};
    RaiiH5Hid const space_id(::H5Screate_simple(1, dims.data(), nullptr), ::H5Sclose);

    hid_t const type_id = H5T_NATIVE_INT;

    RaiiH5Hid const attr_id(
            ::H5Acreate2(*object_id, attribute_name, type_id, *space_id, H5P_DEFAULT, H5P_DEFAULT),
            ::H5Aclose);

    h5check(::H5Awrite(*attr_id, type_id, attribute_value.data()));
}

void write_attribute(
        RaiiH5Hid const& object_id,
        char const* const attribute_name,
        std::span<double const> const& attribute_value)
{
    std::array<hsize_t, 1> const dims {attribute_value.size()};
    RaiiH5Hid const space_id(::H5Screate_simple(1, dims.data(), nullptr), ::H5Sclose);

    hid_t const type_id = H5T_NATIVE_DOUBLE;

    RaiiH5Hid const attr_id(
            ::H5Acreate2(*object_id, attribute_name, type_id, *space_id, H5P_DEFAULT, H5P_DEFAULT),
            ::H5Aclose);

    h5check(::H5Awrite(*attr_id, type_id, attribute_value.data()));
}

void write_celldata(
        RaiiH5Hid const& file_id,
        char const* const dataset_name,
        Kokkos::mdspan<double const, Kokkos::dextents<int, 3>, Kokkos::layout_left> const& data)
{
    // Horrible transposition on the fly!!!
    std::array const
            dims {static_cast<hsize_t>(data.extent(2)),
                  static_cast<hsize_t>(data.extent(1)),
                  static_cast<hsize_t>(data.extent(0))};
    RaiiH5Hid const space_id(::H5Screate_simple(dims.size(), dims.data(), nullptr), ::H5Sclose);

    hid_t const type_id = H5T_NATIVE_DOUBLE;

    RaiiH5Hid const
            dset(::H5Dcreate2(
                         *file_id,
                         dataset_name,
                         type_id,
                         *space_id,
                         H5P_DEFAULT,
                         H5P_DEFAULT,
                         H5P_DEFAULT),
                 ::H5Dclose);

    write_attribute(dset, "Attribute", "Scalars", H5T_CSET_ASCII);
    h5check(::H5Dwrite(*dset, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data_handle()));
}

} // namespace

void write_imagedata(
        char const* const filename,
        char const* const dataset_name,
        Kokkos::mdspan<double const, Kokkos::dextents<int, 3>, Kokkos::layout_left> const& mds)
{
    RaiiH5Hid const
            file_id(::H5Fcreate(filename, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT), ::H5Fclose);
    RaiiH5Hid const vtkhdf_group(
            ::H5Gcreate2(*file_id, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            ::H5Gclose);

    std::array const version {2, 6};
    std::string_view const type("ImageData");
    std::array const whole_extent {0, mds.extent(0), 0, mds.extent(1), 0, mds.extent(2)};
    std::array const origin {0., 0., 0.};
    std::array const spacing {1., 1., 1.};
    std::array const direction {1, 0, 0, 0, 1, 0, 0, 0, 1};

    write_attribute(vtkhdf_group, "Version", version);
    write_attribute(vtkhdf_group, "Type", type, H5T_CSET_ASCII);
    write_attribute(vtkhdf_group, "WholeExtent", whole_extent);
    write_attribute(vtkhdf_group, "Origin", origin);
    write_attribute(vtkhdf_group, "Spacing", spacing);
    write_attribute(vtkhdf_group, "Direction", direction);

    RaiiH5Hid const vtkhdf_celldata_group(
            ::H5Gcreate2(*vtkhdf_group, "CellData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            ::H5Gclose);
    write_celldata(vtkhdf_celldata_group, dataset_name, mds);
}
