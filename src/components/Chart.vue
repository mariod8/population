<template>
  <div class="flex flex-col items-center justify-center w-full">
    <span class="py-4 font-bold">{{ location }} population in {{ year }}</span>
    <div class="flex flex-col [&>div>*]:h-[13px] gap-[4px] items-center">
      <div v-for="data in graphData" class="flex flex-row">
        <span
          class="font-extrabold text-[.6rem] items-center justify-center inline-flex mr-[6px]"
          >{{ formatNumber(data.males.amount) }}</span
        >
        <div
          :style="{
            width: `${calcBarWidth(data.males.ratio)}px`,
            backgroundSize: `${barWidth}px 100%`
          }"
          class="min-w-[2px] rounded-full bg-gradient-to-r from-grad1-1 to-grad1-2 float-left scale-[-1]"
        ></div>
        <span class="font-bold text-[.55rem] items-center justify-between inline-flex w-[50px]">
          {{ data.ageRange.start }}-{{ data.ageRange.end }}
        </span>
        <div
          :style="{
            width: `${calcBarWidth(data.females.ratio)}px`,
            backgroundSize: `${barWidth}px 100%`
          }"
          class="min-w-[2px] rounded-full bg-gradient-to-r from-grad1-1 to-grad1-2"
        ></div>
        <span
          class="font-extrabold text-[.6rem] items-center justify-center inline-flex float-right"
          >{{ formatNumber(data.females.amount) }}</span
        >
      </div>
    </div>
  </div>
</template>
<script lang="ts">
import type { Data, LocationData } from "@/types"
import { formatNumber } from "@/utils"
import type { PropType } from "vue"

const ageGrouping = 3

export default {
  data() {
    return {
      chartWidth: 400,
      barWidth: 100
    }
  },
  methods: {
    formatNumber,
    population() {
      return [
        ...this.data[this.location].info[this.year].males,
        ...this.data[this.location].info[this.year].females
      ].reduce((sum, current) => sum + current, 0)
    },
    populationPerAge() {
      return this.data[this.location].info[this.year].males.map((value, i) => {
        return value + this.data[this.location].info[this.year].females[i]
      })
    },
    maxGenderValue() {
      return Math.max(
        ...[
          ...this.data[this.location].info[this.year].males,
          ...this.data[this.location].info[this.year].females
        ]
      )
    },
    percentage() {
      const population = this.population()

      return this.populationPerAge().map((value) => {
        const percent = value / population
        const product = percent < 0.001 ? 10000 : 1000
        const divisor = product === 10000 ? 100 : 10

        return Math.round(percent * product) / divisor
      })
    },
    calcBarWidth(percent: number) {
      return this.barWidth * percent
    }
  },
  computed: {
    graphData() {
      const max = this.maxGenderValue()
      var age = 0

      return this.percentage().map((value, i) => {
        const data = {
          percentage: value,
          ageRange: { start: age, end: age + ageGrouping - 1 },
          males: {
            amount: this.data[this.location].info[this.year].males[i],
            ratio: this.data[this.location].info[this.year].males[i] / max
          },
          females: {
            amount: this.data[this.location].info[this.year].females[i],
            ratio: this.data[this.location].info[this.year].females[i] / max
          }
        }
        
        age += ageGrouping
        return data
      })
    }
  },
  props: {
    data: {
      type: Object as PropType<Data>,
      required: true
    },
    location: {
      type: String as PropType<keyof Data>,
      required: true
    },
    year: {
      type: Number as PropType<keyof LocationData>,
      required: true
    }
  }
}
</script>
