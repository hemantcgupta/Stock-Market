import React, { Component } from "react";
import SurplusDeficit from './SurplusDeficit';
import BreakEvenAVGFareLoadFactor from './BreakEvenAvgLoadFactor';
import CaskRask from './CaskRask';
import ForexFuelPrice from './ForexFuelPrice';
import CardsRP from './CardsRP';
import APIServices from '../../../../API/apiservices';

const apiServices = new APIServices();

class Graphs extends Component {
  constructor(props) {
    super(props);
    this.state = {
    }
  }

  renderSurplusDeficit = () => {
    return (
      <SurplusDeficit
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        routeGroup={this.props.routeGroup}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        routeId={this.props.routeId}
        {...this.props}
      />
    )
  }

  renderBreakEvenAVGFareLoadFactor = () => {
    return (
      <BreakEvenAVGFareLoadFactor
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        routeGroup={this.props.routeGroup}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        routeId={this.props.routeId}
        {...this.props}
      />
    )
  }

  renderCaskRask = () => {
    return (
      <CaskRask
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        routeGroup={this.props.routeGroup}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        routeId={this.props.routeId}
        {...this.props}
      />
    )
  }

  renderForexFuelPrice = () => {
    return (
      <ForexFuelPrice
        startDate={this.props.startDate}
        endDate={this.props.endDate}
        routeGroup={this.props.routeGroup}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        routeId={this.props.routeId}
        {...this.props}
      />
    )
  }

  renderCardsRP = () => {
    return (
      <CardsRP
        routeGroup={this.props.routeGroup}
        regionId={this.props.regionId}
        countryId={this.props.countryId}
        routeId={this.props.routeId}
        {...this.props}
      />
    )
  }

  render() {
    return (
      <div className="row graphs">

        <div className="col-md-4 col-sm-4 col-xs-12 graph">
          {this.renderSurplusDeficit()}
          {this.renderBreakEvenAVGFareLoadFactor()}
        </div>

        <div className="col-md-4 col-sm-4 col-xs-12 graph">
          {this.renderCaskRask()}
          {this.renderForexFuelPrice()}
        </div>

        {this.renderCardsRP()}

      </div>

    )
  }

}

export default Graphs;