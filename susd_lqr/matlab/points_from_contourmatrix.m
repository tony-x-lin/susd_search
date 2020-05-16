function[xpoints, ypoints, per_point_levels] = points_from_contourmatrix(C)
[~,c] = size(C);

offsets = [];
levels = [];
offset = 1;
while offset +1 < c
    level = C(1,offset);
    len = C(2,offset);
    offsets = [offsets, offset];
    levels = [levels, level];
    offset = offset + len + 1;
end
offsets = [offsets, c];


xpoints = [];
ypoints = [];
per_point_levels = [];
for i = 1:length(offsets)-1
    xpoints = [xpoints, C(1,offsets(i)+1:offsets(i+1)-1), NaN];
    ypoints = [ypoints, C(2,offsets(i)+1:offsets(i+1)-1), NaN];
    per_point_levels = [per_point_levels, levels(i)*ones(1,offsets(i+1)-offsets(i))];
end

return 