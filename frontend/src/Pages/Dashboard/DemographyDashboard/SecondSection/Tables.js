import React, { Component } from "react";
import EnrichTierTable from './enrichTier';
import TopNationality from './top10Nationality';
import AdvanceFlightPurchase from './advanceFlightTable';
import '../DemographyDashboard.scss';

class Tables extends Component {

  render() {
    return (
      <div className="x_panel">
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <EnrichTierTable
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            cityId={this.props.cityId}
            {...this.props} />
        </div>
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <TopNationality
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            cityId={this.props.cityId}
            posDashboard={true}
            {...this.props} />
        </div>
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <AdvanceFlightPurchase
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            cityId={this.props.cityId}
            {...this.props} />
        </div>
      </div >
    )
  }
}
export default Tables;