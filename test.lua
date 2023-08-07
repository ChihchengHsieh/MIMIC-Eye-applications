function()
    
    local function round(num, idp)
        local mult = 10 ^ (idp or 0);
        return math.floor(num * mult + 0.5) / mult;
    end
    -- Crit reduction from Defense
    local defCrit = GetDodgeBlockParryChanceFromDefense()
    -- Total added defense skill from gear
    local defenseGear = round(GetCombatRatingBonus(2), 2)
    -- Calculating Diminishing Returns for Miss
    local defenseEffect =  ((defenseGear*0.04)*16)/((0.956*16)+(defenseGear*0.04))
    -- Total miss after DR
    local miss =  max(0, 5 + defenseEffect)
    local dodge = round(GetDodgeChance(), 2)
    local parry = round(GetParryChance(), 2)
    local block = round(GetBlockChance(), 2)
    local bv = round(GetShieldBlock(), 2)
    local avoidance = round((miss + dodge + parry), 2)
    local resil = GetCombatRatingBonus(15)
    local crit = round(defCrit + resil, 2)
    local ct = round(avoidance + block, 2)
    local critcolor = "|cffff0000"..crit.."|r"
    local ctcolor = "|cffffff00"..ct.."|r"
    local apBase, apPlus, apNeg = UnitAttackPower("player");
    local holy = GetSpellBonusDamage(2)
    local apBase, apPlus, apNeg = UnitAttackPower("player");
    local mhit = GetCombatRatingBonus(6) + GetHitModifier()
    local exp = GetExpertise()
    local baseArmor, effectiveArmor, armor, posBuff, negBuff = UnitArmor("player")
    -- Damage reduction from armor
    local dr = (effectiveArmor / ((83 * 467.5) + effectiveArmor - 22167.5))
    -- Adding Improved righteous fury to DR    
    local RF = WeakAuras.GetActiveTriggers(aura_env.id)[4]
    if RF == true then
        dr = 1-(1-dr)*(1-(.06));
    end
    --Adding BoS to DR
    local BoS = WeakAuras.GetActiveTriggers(aura_env.id)[3]
    if BoS == true then
        dr = 1-(1-dr)*(1-(.03));parry
    end
    --Adding Shield of the Templar to DR
    local templar = WeakAuras.GetActiveTriggers(aura_env.id)[7]
    if templar == true then
        dr = 1-(1-dr)*(1-(.03));
    end
    --Adding DP Glyph to DR
    local DPglyph = WeakAuras.GetActiveTriggers(aura_env.id)[5]
    local DPactive = WeakAuras.GetActiveTriggers(aura_env.id)[6]
    if (DPglyph and DPactive) == true then
        dr = 1-(1-dr)*(1-(.03));
    end
    local max_health = UnitHealthMax("player")
    local eh = max_health / (1 - dr )
    dr = round(min(dr,0.75)*100,4);
    
    
    if crit >= 5.6 then
        critcolor = "|cff00ff00"..crit.."|r"
    end
    
    if ct >= 102.4 then
        ctcolor = "|cff00ff00"..ct.."|r"
    end
    
    if mhit <= 7.5 then
        hitcolor = "|cffff0000"..string.format("%.2f", mhit).."|r"
    elseif mhit >= 7.5 and mhit < 8 then
        hitcolor = "|cffffff00"..string.format("%.2f", mhit).."|r"
    elseif mhit >= 8 and mhit < 8.1 then
        hitcolor = "|cff00ff00"..string.format("%.2f", mhit).."|r"
    elseif mhit >= 8.1 then
        hitcolor = "|cff00ccff"..string.format("%.2f", mhit).."|r"
    end
    
    if exp <= 25.99 then
        expertisecolor = "|cffff0000"..string.format("%.0f", exp).."|r"
    elseif exp >= 26 and exp < 27 then
        expertisecolor = "|cffffff00"..string.format("%.0f", exp).."|r"
    elseif exp >= 27 and exp < 56 then
        expertisecolor = "|cff00ff00"..string.format("%.0f", exp).."|r"
    elseif exp >= 56 and exp < 57 then
        expertisecolor = "|cffDA70D6"..string.format("%.0f", exp).."|r"
    elseif exp > 56 then
        expertisecolor = "|cff00ccff"..string.format("%.0f", exp).."|r"
    end
    
    return         
    "|cff00ccff"..string.format("%.2f", miss).."%|r",
    "|cff00ccff"..string.format("%.2f", dodge).."%|r",
    "|cff00ccff"..string.format("%.2f", parry).."%|r",
    "|cffDA70D6"..string.format("%.2f", avoidance).."%|r",
    "|cff00E5EE"..string.format("%.2f", block).."%|r",
    critcolor,
    ctcolor,
    "|cffFF4500"..string.format("%.0f", bv).."|r",
    "|cffffff00"..string.format("%.0f", holy).."|r",
    "|cffFF4500"..string.format("%.0f", effectiveArmor).."|r",
    "|cffFF4500"..string.format("%.2f", dr).."%|r",
    "|cffFF4500"..string.format("%.0f", eh).."|r",
    "|cffFF4500"..string.format("%.0f", max_health).."|r",
    hitcolor,
    expertisecolor,
    "|cffffff00"..string.format("%.0f", apBase + apPlus + apNeg).."|r"
    
end




function ()
    local block = GetBlockChance()
    return "Block: " .. string.format("%.2f", block) .. "%"
end