import React, { Component } from "react";
import ChannelWisePerform from './channelwiseperform';
import AnciallaryItems from './AnciallaryItems';
import MarketShare from './MarketShare';
import '../PosDashboard.scss';

class Tables extends Component {

  render() {
    return (
      <div className="x_panel">
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <ChannelWisePerform
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            cityId={this.props.cityId}
            {...this.props} />
        </div>
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <AnciallaryItems
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            cityId={this.props.cityId}
            posDashboard={true}
            {...this.props} />
        </div>
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <MarketShare
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