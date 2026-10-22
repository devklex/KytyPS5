#include "graphics/shader/shaderVertexMetadata.h"
#include "graphics/host_gpu/renderer/meshDrawChunks.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "ShaderVertexMetadataTests: failed: %s\n", text);
		std::abort();
	}
}

struct Fixture {
	std::array<uint16_t, static_cast<size_t>(AgcDirectResourceType::Last) + 1> offsets {};
	ShaderUserData                                                             user_data {};
	ShaderSemantic                                                             semantic {};
	ShaderMappedData                                                           mapped {};

	Fixture() {
		offsets.fill(AGC_ILLEGAL_DIRECT_OFFSET);
		offsets[static_cast<size_t>(AgcDirectResourceType::PtrVertexBufferTable)]     = 2;
		offsets[static_cast<size_t>(AgcDirectResourceType::PtrVertexAttribDescTable)] = 4;
		user_data.direct_resource_offset = offsets.data();
		user_data.direct_resource_count  = static_cast<uint16_t>(offsets.size());
		mapped.user_data                 = &user_data;
		mapped.input_semantics           = &semantic;
		mapped.num_input_semantics       = 1;
	}
};

void CheckRejected(const ShaderMappedData& data, const char* text) {
	ShaderVertexMetadata output;
	std::array<ShaderSemantic, 7> prior_semantics {};
	output.vertex_buffer_reg = 37;
	output.vertex_attrib_reg = 41;
	output.input_semantics   = prior_semantics;
	std::string error;
	Check(!ShaderReadVertexMetadata(data, 64, output, &error), text);
	Check(!error.empty(), "metadata rejection omitted its diagnostic");
	Check(output.vertex_buffer_reg == 37 && output.vertex_attrib_reg == 41 &&
	          output.input_semantics.data() == prior_semantics.data() &&
	          output.input_semantics.size() == prior_semantics.size(),
	      "metadata rejection changed the prior output");
}

void TestValidAndInvalidMetadata() {
	Fixture              fixture;
	ShaderVertexMetadata output;
	std::string          error;
	Check(ShaderReadVertexMetadata(fixture.mapped, 64, output, &error),
	      "valid AGC vertex metadata was rejected");
	Check(output.vertex_buffer_reg == 2 && output.vertex_attrib_reg == 4 &&
	          output.input_semantics.size() == 1,
	      "valid AGC vertex metadata was decoded incorrectly");

	Fixture buffer_only;
	buffer_only.offsets[static_cast<size_t>(AgcDirectResourceType::PtrVertexAttribDescTable)] =
	    AGC_ILLEGAL_DIRECT_OFFSET;
	buffer_only.mapped.input_semantics     = nullptr;
	buffer_only.mapped.num_input_semantics = 0;
	Check(ShaderReadVertexMetadata(buffer_only.mapped, 64, output, &error),
	      "vertex-buffer-only AGC metadata was rejected");
	Check(output.vertex_buffer_reg == 2 && output.vertex_attrib_reg == -1 &&
	          output.input_semantics.empty(),
	      "vertex-buffer-only AGC metadata was decoded incorrectly");

	Fixture attrib_only;
	attrib_only.offsets[static_cast<size_t>(AgcDirectResourceType::PtrVertexBufferTable)] =
	    AGC_ILLEGAL_DIRECT_OFFSET;
	CheckRejected(attrib_only.mapped, "vertex-attribute-only AGC metadata was accepted");

	auto missing_header      = fixture.mapped;
	missing_header.user_data = nullptr;
	CheckRejected(missing_header, "missing AGC user-data header was accepted");

	Fixture missing_offsets;
	missing_offsets.user_data.direct_resource_offset = nullptr;
	CheckRejected(missing_offsets.mapped, "missing direct-resource offsets were accepted");

	Fixture excessive_resources;
	excessive_resources.user_data.direct_resource_count =
	    static_cast<uint16_t>(excessive_resources.offsets.size() + 1);
	CheckRejected(excessive_resources.mapped, "excessive direct-resource count was accepted");

	Fixture excessive_semantics;
	excessive_semantics.mapped.num_input_semantics = ShaderVertexInputInfo::RES_MAX + 1;
	CheckRejected(excessive_semantics.mapped, "excessive vertex semantic count was accepted");

	Fixture excessive_register;
	excessive_register.offsets[static_cast<size_t>(AgcDirectResourceType::PtrVertexBufferTable)] =
	    63;
	CheckRejected(excessive_register.mapped, "out-of-domain vertex table SGPR was accepted");

	Fixture excessive_attrib_register;
	excessive_attrib_register
	    .offsets[static_cast<size_t>(AgcDirectResourceType::PtrVertexAttribDescTable)] = 63;
	CheckRejected(excessive_attrib_register.mapped,
	              "out-of-domain vertex attribute table SGPR was accepted");

	Fixture missing_semantics;
	missing_semantics.mapped.input_semantics = nullptr;
	CheckRejected(missing_semantics.mapped, "missing vertex semantic array was accepted");
}

// NVIDIA's limits: maxMeshWorkGroupCount {4194304, 65535, 65535}, maxMeshWorkGroupTotalCount 4194304.
// Astro Bot's Sky Garden draws one mesh group of 71682 instances, which one dispatch cannot carry.
void TestMeshDrawInstanceChunks() {
	constexpr uint32_t max_x = 4194304, max_y = 65535, max_total = 4194304;
	const auto covered = [&](uint32_t groups, uint32_t instances, uint32_t expected_dispatches) {
		const auto per_dispatch = MeshDrawInstancesPerDispatch(groups, max_x, max_y, max_total);
		Check(per_dispatch != 0, "a representable mesh draw got no instances per dispatch");
		uint32_t dispatches = 0, next = 0;
		for (uint32_t first = 0; first < instances; first += per_dispatch, dispatches++) {
			const auto count = std::min(per_dispatch, instances - first);
			Check(first == next && count <= max_y && uint64_t {groups} * count <= max_total,
			      "a mesh draw dispatch broke a workgroup limit or skipped instances");
			next = first + count;
		}
		Check(next == instances && dispatches == expected_dispatches,
		      "mesh draw dispatches do not cover every instance exactly once");
	};
	covered(1, 71682, 2);       // Sky Garden: 65535 + 6147 instances.
	covered(1, 65535, 1);       // Exactly at the per-dimension limit.
	covered(100, 100000, 3);    // The total-count limit binds first: 41943 per dispatch.
	covered(4194304, 1, 1);     // The whole total count in X leaves one instance per dispatch.
	Check(MeshDrawInstancesPerDispatch(100, max_x, max_y, max_total) == 41943,
	      "the total workgroup count limit was ignored");
	Check(MeshDrawInstancesPerDispatch(4194305, max_x, max_y, max_total) == 0,
	      "a draw with more groups than maxMeshWorkGroupCount[0] was accepted");
	Check(MeshDrawInstancesPerDispatch(0, max_x, max_y, max_total) == 0,
	      "a draw without groups was given instances");
}

} // namespace

int main() {
	TestValidAndInvalidMetadata();
	TestMeshDrawInstanceChunks();
	std::puts("ShaderVertexMetadataTests: all cases passed");
	return 0;
}
